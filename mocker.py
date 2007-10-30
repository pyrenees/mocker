import __builtin__
import inspect
import types
import sys
import os
import gc


__all__ = ["Mocker", "expect", "SAME", "CONTAINS", "ANY", "VARIOUS"]


ERROR_PREFIX = "[Mocker] "


# --------------------------------------------------------------------
# Exceptions

class MatchError(AssertionError):
    """Raised when an unknown expression is seen in playback mode."""


# --------------------------------------------------------------------
# Helper for chained-style calling.

class expect(object):
    """This is a simple helper that allows a different call-style.

    With this class one can comfortably do chaining of calls to the
    mocker object responsible by the object being handler. For instance::

        expect(obj.attr).result(3).count(1, 2)

    Is the same as::

        obj.attr
        mocker.result(3)
        mocker.count(1, 2)

    """

    def __init__(self, mock, attr=None):
        self._mock = mock
        self._attr = attr

    def __getattr__(self, attr):
        return self.__class__(self._mock, attr)

    def __call__(self, *args, **kwargs):
        getattr(self._mock.__mocker__, self._attr)(*args, **kwargs)
        return self


# --------------------------------------------------------------------
# Mocker.

class classinstancemethod(object):

    def __init__(self, method):
        self.method = method

    def __get__(self, obj, cls=None):
        def bound_method(*args, **kwargs):
            return self.method(cls, obj, *args, **kwargs)
        return bound_method


class State(object):

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return self._name


RECORD = State("RECORD")
REPLAY = State("REPLAY")
RESTORE = State("RESTORE")


class MockerBase(object):
    """Controller of mock objects.

    A mocker instance is used to command recording and replay of
    expectations on any number of mock objects.

    Expectations should be expressed for the mock object while in
    record mode (the initial one) by using the mock object itself,
    and using the mocker (and/or C{expect()} as a helper) to define
    additional behavior for each event.  For instance::

        mock = mocker.mock()
        mock.hello()
        mocker.result(10)
        mocker.replay()
        assert mock.hello() == 10
        mock.restore()
        mock.verify()

    In this short excerpt a mock object is being created, then an
    expectation of a call to the C{hello()} method was recorded, and
    when that happens the method should return the value C{10}.  Then,
    the mocker is put in replay mode, and the expectation is satisfied
    by calling the C{hello()} method, which indeed returns 10.  Finally,
    a call to the L{restore()} method is performed to undo any needed
    changes made in the environment, and the L{verify()} method is
    called to ensure that all defined expectations were met.

    The same logic can be expressed more elegantly using the
    C{with mocker:} statement, as follows::

        mock = mocker.mock()
        mock.hello()
        mocker.result(10)
        with mocker:
            assert mock.hello() == 10

    Also, the MockerTestCase class, which integrates the mocker on
    a unittest.TestCase subclass, may be used to reduce the overhead
    of controlling the mocker.  A test could be written as follows::

        class SampleTest(MockerTestCase):

            def test_hello(self):
                mock = self.mocker.mock()
                mock.hello()
                self.mocker.result(10)
                self.mocker.replay()
                assert mock.hello() == 10
    """

    _recorders = []

    # For convenience only.
    on = expect

    class __metaclass__(type):
        def __init__(self, name, bases, dict):
            # Make independent lists on each subclass, inheriting from parent.
            self._recorders = list(getattr(self, "_recorders", ()))

    def __init__(self):
        self._recorders = self._recorders[:]
        self._events = []
        self._state = RECORD
        self._ordering = False
        self._last_orderer = None

    def _set_state(self, state):
        if state is not self._state:
            self._state = state
            for event in self._events:
                event.set_state(state)

    def get_state(self):
        """Return the current state (RECORD, REPLAY or RESTORE).

        The initial state is RECORD.
        """
        return self._state

    def replay(self):
        """Change state to expect recorded events to be reproduced.
        
        An alternative and more comfortable way to replay changes is
        using the 'with' statement, as follows::

            mocker = Mocker()
            <record events>
            with mocker:
                <reproduce events>

        The 'with' statement will automatically put mocker in replay
        mode, and will also verify if all events were correctly reproduced
        at the end (using L{verify()}), and also restore any changes done
        in the environment (with L{restore()}).

        Also check the MockerTestCase class, which integrates the
        unittest.TestCase class with mocker.
        """
        self._set_state(REPLAY)

    def record(self):
        """Put the mocker on recording mode, where expectations are defined.
        
        That's the initial state when the mocker is constructed.
        """
        self._set_state(RECORD)

    def restore(self):
        """Restore all changes done in the environment.

        This should always be called after the test is complete (succeeding
        or not).  There are ways to call this method automatically on
        completion (e.g. using a with mocker: statement, or using the
        MockerTestCase class.
        """
        self._set_state(RESTORE)

    def is_ordering(self):
        """Return true if all events are being ordered.

        See the L{ordered()} method.
        """
        return self._ordering

    def ordered(self):
        """Expect following events to be reproduced in the recorded order.

        By default, mocker won't force events to happen precisely in
        the order they were recorded.  Calling this method will change
        this behavior so that events will only match if reproduced in
        the correct order.

        Running the L{unordered()} method will put the mocker back on
        unordered mode.

        This method may also be used with the 'with' statement, like so::

            with mocker.ordered():
                <record events>

        In this case, only expressions in <record events> will be ordered,
        and the mocker will be back in unordered mode after the 'with' block.
        """
        self._ordering = True
        return OrderedContext(self)

    def unordered(self):
        """Don't expect following events to be reproduce in the recorded order.

        This will undo the effect of the L{ordered()} method, putting the
        mocker back in the default unordered mode.
        """
        self._ordering = False
        self._last_orderer = None

    def get_events(self):
        """Return all recorded events."""
        return self._events[:]

    def add_event(self, event):
        """Add an event.

        This method is used internally by the implementation, and
        shouldn't be needed on normal mocker usage.
        """
        self._events.append(event)
        if self._ordering:
            orderer = event.add_task(Orderer())
            if self._last_orderer:
                orderer.add_dependency(self._last_orderer)
            self._last_orderer = orderer
        return event

    def verify(self):
        """Check if all expectations were met, and raise AssertionError if not.

        The exception message will include a nice description of which
        expectations were not met, and why.
        """
        errors = []
        for event in self._events:
            try:
                event.verify()
            except AssertionError, e:
                error = str(e)
                if not error:
                    raise RuntimeError("Empty error message from %r"
                                       % event)
                errors.append(error)
        if errors:
            message = [ERROR_PREFIX + "Unmet expectations:", ""]
            for error in errors:
                lines = error.splitlines()
                message.append("=> " + lines.pop(0))
                message.extend(" " + line for line in lines)
                message.append("")
            raise AssertionError(os.linesep.join(message))

    def mock(self, spec_and_type=None, spec=None, type=None, name=None):
        """Return a new mock object.

        @param spec_and_type: Handy positional argument which sets both
                     spec and type.
        @param spec: Method calls will be checked for correctness against
                     the given class.
        @param type: If set, the Mock's __class__ attribute will return
                     the given type.  This will make C{isinstance()} calls
                     on the object work.
        @param name: Name for the mock object, used in the representation of
                     expressions.  The name is rarely needed, as it's usually
                     guessed correctly from the variable name used.
        """
        if spec_and_type is not None:
            spec = type = spec_and_type
        return Mock(self, spec=spec, type=type, name=name)

    def proxy(self, object, spec=True, type=True, name=None, passthrough=True):
        """Return a new mock object which proxies to the given object.
 
        Proxies are useful when only part of the behavior of an object
        is to be mocked.  Unknown expressions may be passed through to
        the real implementation implicitly (if the C{passthrough} argument
        is True), or explicitly (using the L{passthrough()} method
        on the event).

        @param object: Real object to be proxied.
        @param spec: Method calls will be checked for correctness against
                     the given object, which may be a class or an instance
                     where attributes will be looked up.  Defaults to the
                     the C{object} parameter.  May be set to None explicitly,
                     in which case spec checking is disabled.  Checks may
                     also be disabled explicitly on a per-event basis with
                     the L{nospec()} method.
        @param type: If set, the Mock's __class__ attribute will return
                     the given type.  This will make C{isinstance()} calls
                     on the object work.  Defaults to the type of the
                     C{object} parameter.  May be set to None explicitly.
        @param name: Name for the mock object, used in the representation of
                     expressions.  The name is rarely needed, as it's usually
                     guessed correctly from the variable name used.
        @param passthrough: If set to False, passthrough of actions on the
                            proxy to the real object will only happen when
                            explicitly requested via the L{passthrough()}
                            method.
        """
        if spec is True:
            spec = object
        if type is True:
            type = __builtin__.type(object)
        return Mock(self, spec=spec, type=type, object=object,
                    name=name, passthrough=passthrough)

    def replace(self, object, spec=True, type=True, name=None,
                passthrough=True):
        """Create a proxy, and replace the original object with the mock.

        On replay, the original object will be replaced by the returned
        proxy in all dictionaries found in the running interpreter via
        the garbage collecting system.  This should cover module
        namespaces, class namespaces, instance namespaces, and so on.

        @param object: Real object to be proxied, and replaced by the mock
                       on replay mode.
        @param spec: Method calls will be checked for correctness against
                     the given object, which may be a class or an instance
                     where attributes will be looked up.  Defaults to the
                     the C{object} parameter.  May be set to None explicitly,
                     in which case spec checking is disabled.  Checks may
                     also be disabled explicitly on a per-event basis with
                     the L{nospec()} method.
        @param type: If set, the Mock's __class__ attribute will return
                     the given type.  This will make C{isinstance()} calls
                     on the object work.  Defaults to the type of the
                     C{object} parameter.  May be set to None explicitly.
        @param name: Name for the mock object, used in the representation of
                     expressions.  The name is rarely needed, as it's usually
                     guessed correctly from the variable name used.
        @param passthrough: If set to False, passthrough of actions on the
                            proxy to the real object will only happen when
                            explicitly requested via the L{passthrough()}
                            method.
        """
        if isinstance(object, basestring):
            if name is None:
                name = object
            import_stack = object.split(".")
            attr_stack = []
            while import_stack:
                module_path = ".".join(import_stack)
                try:
                    object = __import__(module_path, {}, {}, [""])
                except ImportError:
                    attr_stack.insert(0, import_stack.pop())
                    continue
                else:
                    for attr in attr_stack:
                        object = getattr(object, attr)
                    break
        mock = self.proxy(object, spec, type, name, passthrough)
        event = self.add_event(Event())
        event.add_task(ProxyReplacer(mock))
        return mock

    def patch(self, object):
        """Patch an existing object to reproduce recorded events.

        @param object: Class or instance to be patched.

        The result of this method is still a mock object, which can be
        used like any other mock object to record events.  The difference
        is that when the mocker is put on replay mode, the *real* object
        will be modified to behave according to recorded expectations.

        Patching works in individual instances, and also in classes.
        When an instance is patched, recorded events will only be
        considered on this specific instance, and other instances should
        behave normally.  When a class is patched, the reproduction of
        events will be considered on any instance of this class once
        created (collectively).

        Observe that, unlike with proxies which catch only events done
        through the mock object, *all* accesses to recorded expectations
        will be considered;  even these coming from the object itself
        (e.g. C{self.hello()} is considered if this method was patched).
        While this is a very powerful feature, and many times the reason
        to use patches in the first place, it's important to keep this
        behavior in mind.

        Patching of the original object only takes place when the mocker
        is put on replay mode, and the patched object will be restored
        to its original state once the L{restore()} method is called
        (explicitly, or implicitly with alternative conventions, such as
        a C{with mocker:} block, or a MockerTestCase class).
        """
        patcher = Patcher()
        event = self.add_event(Event())
        event.add_task(patcher)
        mock = Mock(self, object=object, patcher=patcher, passthrough=True)
        object.__mocker_mock__ = mock
        return mock

    def act(self, path):
        """This is called by mock objects whenever something happens to them.

        This method is part of the implementation between the mocker
        and mock objects.
        """
        if self._state is RECORD:
            event = self.add_event(Event(path))
            for recorder in self._recorders:
                recorder(self, event)
            return Mock(self, path)
        elif self._state is REPLAY:
            for event in sorted(self._events, key=lambda e:e.satisfied()):
                if event.matches(path):
                    return event.run(path)
            raise MatchError(ERROR_PREFIX + "Unexpected expression: %s" % path)
            # XXX Show the full state here.
        else:
            raise RuntimeError(ERROR_PREFIX + "Not in record or playback mode")

    @classinstancemethod
    def get_recorders(cls, self):
        """Return recorders associated with this mocker class or instance.

        This method may be called on mocker instances and also on mocker
        classes.  See the L{add_recorder()} method for more information.
        """
        return (self or cls)._recorders[:]

    @classinstancemethod
    def add_recorder(cls, self, recorder):
        """Add a recorder to this mocker class or instance.

        @param recorder: Callable accepting C{(mocker, event)} as parameters.

        This is part of the implementation of mocker.

        All registered recorders are called for translating events that
        happen during recording into expectations to be met once the state
        is switched to replay mode.

        This method may be called on mocker instances and also on mocker
        classes.  When called on a class, the recorder will be used by
        all instances, and also inherited on subclassing.  When called on
        instances, the recorder is added only to the given instance.
        """
        (self or cls)._recorders.append(recorder)
        return recorder

    @classinstancemethod
    def remove_recorder(cls, self, recorder):
        """Remove the given recorder from this mocker class or instance.

        This method may be called on mocker classes and also on mocker
        instances.  See the L{add_recorder()} method for more information.
        """
        (self or cls)._recorders.remove(recorder)

    def result(self, value):
        """Make the last recorded event return the given value on replay.
        
        @param value: Object to be returned when the event is replayed.
        """
        self.call(lambda *args, **kwargs: value)

    def throw(self, exception):
        """Make the last recorded event raise the given exception on replay.

        @param exception: Class or instance of exception to be raised.
        """
        def raise_exception(*args, **kwargs):
            raise exception
        self.call(raise_exception)

    def call(self, func):
        """Make the last recorded event cause the given function to be called.

        @param func: Function to be called.

        The result of the function will be used as the event result.
        """
        self._events[-1].add_task(FunctionRunner(func))

    def count(self, min, max=False):
        """Last recorded event must be replayed between min and max times.

        @param min: Minimum number of times that the event must happen.
        @param max: Maximum number of times that the event must happen.  If
                    not given, it defaults to the same value of the C{min}
                    parameter.  If set to None, there is no upper limit, and
                    the expectation is met as long as it happens at least
                    C{min} times.
        """
        event = self._events[-1]
        for task in event.get_tasks():
            if isinstance(task, RunCounter):
                event.remove_task(task)
        event.add_task(RunCounter(min, max))

    def order(self, *path_holders):
        """Ensure that events referred to by given objects happen in order.

        As an example::

            mock = mocker.mock()
            expr1 = mock.hello()
            expr2 = mock.world
            expr3 = mock.x.y.z
            mocker.order(expr1, expr2, expr3)

        This method of ordering only works when the expression returns
        another object.  For other methods of ordering check the
        L{ordered()}, L{after()}, and L{before()} methods.

        @param path_holders: Objects returned as the result of recorded events.
        """
        last_orderer = None
        for path_holder in path_holders:
            if type(path_holder) is Path:
                path = path_holder
            else:
                path = path_holder.__mocker_path__
            for event in self._events:
                if event.path is path:
                    for task in event.get_tasks():
                        if isinstance(task, Orderer):
                            orderer = task
                            break
                    else:
                        orderer = Orderer()
                        event.add_task(orderer)
                    if last_orderer:
                        orderer.add_dependency(last_orderer)
                    last_orderer = orderer
                    break

    def after(self, *path_holders):
        """Last recorded event must happen after events referred to.

        @param path_holders: Objects returned as the result of recorded events.

        See L{order()} for more information.
        """
        last_path = self._events[-1].path
        for path_holder in path_holders:
            self.order(path_holder, last_path)

    def before(self, *path_holders):
        """Last recorded event must happen before events referred to.

        @param path_holders: Objects returned as the result of recorded events.

        See L{order()} for more information.
        """
        last_path = self._events[-1].path
        for path_holder in path_holders:
            self.order(last_path, path_holder)

    def nospec(self):
        """Don't check method specification of real object on last event."""
        event = self._events[-1]
        for task in event.get_tasks():
            if isinstance(task, SpecChecker):
                event.remove_task(task)

    def passthrough(self):
        """Make the last recorder event act on the real object once seen.

        This can only be used on proxies, as returned by the L{proxy()}
        and L{replace()} methods, or on patched objects (L{patch()}).
        """
        event = self._events[-1]
        if event.path.root_object is None:
            raise TypeError("Mock object isn't a proxy")
        event.add_task(PathExecuter())

    def __enter__(self):
        """Enter in a 'with' context.  This will run replay()."""
        self.replay()
        return self

    def __exit__(self, type, value, traceback):
        """Exit from a 'with' context.

        This will run restore() at all times, but will only run verify()
        if the 'with' block itself hasn't raised an exception.  Exceptions
        in that block are never swallowed.
        """
        self.restore()
        if type is None:
            self.verify()
        return False


class OrderedContext(object):

    def __init__(self, mocker):
        self._mocker = mocker

    def __enter__(self):
        return None

    def __exit__(self, type, value, traceback):
        self._mocker.unordered()


class Mocker(MockerBase):
    __doc__ = MockerBase.__doc__

# Decorator to add recorders on the standard Mocker class.
recorder = Mocker.add_recorder


# --------------------------------------------------------------------
# Mock object.

class Mock(object):

    def __init__(self, mocker, path=None, name=None, spec=None, type=None,
                 object=None, passthrough=False, patcher=None):
        self.__mocker__ = mocker
        self.__mocker_path__ = path or Path(self, object)
        self.__mocker_name__ = name
        self.__mocker_spec__ = spec
        self.__mocker_object__ = object
        self.__mocker_passthrough__ = passthrough
        self.__mocker_patcher__ = patcher
        self.__mocker_replace__ = False
        self.__mocker_type__ = type

    def __mocker_act__(self, kind, args=(), kwargs={}, object=None):
        if self.__mocker_name__ is None:
            self.__mocker_name__ = find_object_name(self, 2)
        action = Action(kind, args, kwargs, self.__mocker_path__)
        path = self.__mocker_path__ + action
        if object is not None:
            path.root_object = object
        try:
            return self.__mocker__.act(path)
        except MatchError, exception:
            if (self.__mocker_type__ is not None and
                kind == "getattr" and args == ("__class__",)):
                return self.__mocker_type__
            root_mock = path.root_mock
            if (path.root_object is not None and
                root_mock.__mocker_passthrough__):
                return path.execute(path.root_object)
            # Reinstantiate to show raise statement on traceback, and
            # also to make it shorter.
            raise MatchError(str(exception))
        except AssertionError, e:
            lines = str(e).splitlines()
            message = [ERROR_PREFIX + "Unmet expectation:", ""]
            message.append("=> " + lines.pop(0))
            message.extend(" " + line for line in lines)
            message.append("")
            raise AssertionError(os.linesep.join(message))

    def __getattribute__(self, name):
        if name.startswith("__mocker_"):
            return super(Mock, self).__getattribute__(name)
        return self.__mocker_act__("getattr", (name,))

    def __call__(self, *args, **kwargs):
        return self.__mocker_act__("call", args, kwargs)


def find_object_name(obj, depth=0):
    """Try to detect how the object is named on a previous scope."""
    try:
        frame = sys._getframe(depth+1)
    except:
        return None
    for name, frame_obj in frame.f_locals.iteritems():
        if frame_obj is obj:
            return name
    self = frame.f_locals.get("self")
    if self is not None:
        try:
            items = list(self.__dict__.iteritems())
        except:
            pass
        else:
            for name, self_obj in items:
                if self_obj is obj:
                    return name
    return None


# --------------------------------------------------------------------
# Action and path.

class Action(object):

    def __init__(self, kind, args, kwargs, path=None):
        self.kind = kind
        self.args = args
        self.kwargs = kwargs
        self.path = path
        self._execute_cache = {}

    def __repr__(self):
        if self.path is None:
            return "Action(%r, %r, %r)" % (self.kind, self.args, self.kwargs)
        return "Action(%r, %r, %r, %r)" % \
               (self.kind, self.args, self.kwargs, self.path)

    def __eq__(self, other):
        return (self.kind == other.kind and
                self.args == other.args and
                self.kwargs == other.kwargs)

    def __ne__(self, other):
        return not self.__eq__(other)

    def matches(self, other):
        return (self.kind == other.kind and
                match_params(self.args, self.kwargs, other.args, other.kwargs))

    def execute(self, object):
        # This caching scheme may fail if the object gets deallocated before
        # the action, as the id might get reused.  It's somewhat easy to fix
        # that with a weakref callback.  For our uses, though, the object
        # should never get deallocated before the action itself, so we'll
        # just keep it simple.
        if id(object) in self._execute_cache:
            return self._execute_cache[id(object)]
        execute = getattr(object, "__mocker_execute__", None)
        if execute is not None:
            result = execute(self, object)
        else:
            kind = self.kind
            if kind == "getattr":
                result = getattr(object, self.args[0])
            elif kind == "call":
                result = object(*self.args, **self.kwargs)
            else:
                raise RuntimeError("Don't know how to apply %r kind"
                                   % self.kind)
        self._execute_cache[id(object)] = result
        return result


class Path(object):

    def __init__(self, root_mock, root_object=None, actions=()):
        self.root_mock = root_mock
        self.root_object = root_object
        self.actions = tuple(actions)
        self.__mocker_replace__ = False

    @property
    def parent_path(self):
        if not self.actions:
            return None
        return self.actions[-1].path

    def __add__(self, action):
        """Return a new path which includes the given action at the end."""
        return self.__class__(self.root_mock, self.root_object,
                              self.actions + (action,))

    def __eq__(self, other):
        """Verify if the two paths are equal.
        
        Two paths are equal if they refer to the same mock object, and
        have the actions with equal kind, args and kwargs.
        """
        if (self.root_mock is not other.root_mock or
            self.root_object is not other.root_object or
            len(self.actions) != len(other.actions)):
            return False
        for action, other_action in zip(self.actions, other.actions):
            if action != other_action:
                return False
        return True

    def matches(self, other):
        """Verify if the two paths are equivalent.
        
        Two paths are equal if they refer to the same mock object, and
        have the same actions performed on them.
        """
        if (self.root_mock is not other.root_mock or
            len(self.actions) != len(other.actions)):
            return False
        for action, other_action in zip(self.actions, other.actions):
            if not action.matches(other_action):
                return False
        return True

    def execute(self, object):
        """Execute all actions sequentially on object, and return result.
        """
        for action in self.actions:
            object = action.execute(object)
        return object

    def __str__(self):
        """Transform the path into a nice string such as obj.x.y('z')."""
        attrs = [self.root_mock.__mocker_name__ or "<mock>"]
        for action in self.actions:
            if action.kind == "getattr":
                attrs.append(action.args[0])
            elif action.kind == "call":
                args = [repr(x) for x in action.args]
                for pair in sorted(action.kwargs.iteritems()):
                    args.append("%s=%r" % pair)
                attrs[-1] += "(%s)" % ", ".join(args)
            else:
                raise RuntimeError("Don't know how to format kind %r" %
                                   action.kind)
        return ".".join(attrs)


class SpecialArgument(object):
    """Base for special arguments for matching parameters."""

    def __init__(self, object=None):
        self.object = object

    def __repr__(self):
        if self.object is None:
            return self.__class__.__name__
        else:
            return "%s(%r)" % (self.__class__.__name__, self.object)

    def matches(self, other):
        return True

    def __eq__(self, other):
        return type(other) == type(self) and self.object == other.object


class ANY(SpecialArgument):
    """Matches any single argument."""

ANY = ANY()


class VARIOUS(SpecialArgument):
    """Matches zero or more arguments."""

VARIOUS = VARIOUS()


class ARGS(SpecialArgument):
    """Matches zero or more positional arguments."""

ARGS = ARGS()


class KWARGS(SpecialArgument):
    """Matches zero or more keyword arguments."""

KWARGS = KWARGS()


class SAME(SpecialArgument):

    def matches(self, other):
        return self.object is other

    def __eq__(self, other):
        return type(other) == type(self) and self.object is other.object


class CONTAINS(SpecialArgument):

    def matches(self, other):
        try:
            other.__contains__
        except AttributeError:
            try:
                iter(other)
            except TypeError:
                # If an object can't be iterated, and has no __contains__
                # hook, it'd blow up on the test below.  We test this in
                # advance to prevent catching more errors than we really
                # want.
                return False
        return self.object in other


def match_params(args1, kwargs1, args2, kwargs2):
    """Match the two sets of parameters, considering the special VARIOUS."""

    has_args = ARGS in args1
    has_kwargs = KWARGS in args1

    if has_kwargs:
        args1 = [arg1 for arg1 in args1 if arg1 is not KWARGS]
    elif len(kwargs1) != len(kwargs2):
        return False

    if not has_args and len(args1) != len(args2):
        return False

    # Either we have the same number of kwargs, or unknown keywords are
    # accepted (KWARGS was used), so check just the ones in kwargs1.
    for key, arg1 in kwargs1.iteritems():
        if key not in kwargs2:
            return False
        arg2 = kwargs2[key]
        if isinstance(arg1, SpecialArgument):
            if not arg1.matches(arg2):
                return False
        elif arg1 != arg2:
            return False

    # Keywords match.  Now either we have the same number of
    # arguments, or ARGS was used.  If ARGS wasn't used, arguments
    # must match one-on-one necessarily.
    if not has_args:
        for arg1, arg2 in zip(args1, args2):
            if isinstance(arg1, SpecialArgument):
                if not arg1.matches(arg2):
                    return False
            elif arg1 != arg2:
                return False
        return True

    # Easy choice. Keywords are matching, and anything on args is accepted.
    if (ARGS,) == args1:
        return True

    # We have something different there. If we don't have positional
    # arguments on the original call, it can't match.
    if not args2:
        # Unless we have just several VARIOUS (which is bizarre, but..).
        for arg1 in args1:
            if arg1 is not ARGS:
                return False
        return True

    # Ok, all bets are lost.  We have to actually do the more expensive
    # matching.  This is an algorithm based on the idea of the Levenshtein
    # Distance between two strings, but heavily hacked for this purpose.
    args2l = len(args2)
    if args1[0] is ARGS:
        args1 = args1[1:]
        array = [0]*args2l
    else:
        array = [1]*args2l
    for i in range(len(args1)):
        last = array[0]
        if args1[i] is ARGS:
            for j in range(1, args2l):
                last, array[j] = array[j], min(array[j-1], array[j], last)
        else:
            array[0] = i or int(args1[i] != args2[0])
            for j in range(1, args2l):
                last, array[j] = array[j], last or int(args1[i] != args2[j])
        if 0 not in array:
            return False
    if array[-1] != 0:
        return False
    return True


# --------------------------------------------------------------------
# Event and task base.

class Event(object):
    """Aggregation of tasks that keep track of a recorded action.

    An event represents something that may or may not happen while the
    mocked environment is running, such as an attribute access, or a
    method call.  The event is composed of several tasks that are
    orchestrated together to create a composed meaning for the event,
    including for which actions it should be run, what happens when it
    runs, and what's the expectations about the actions run.
    """

    def __init__(self, path=None):
        self.path = path
        self._tasks = []

    def add_task(self, task):
        """Add a new task to this taks."""
        self._tasks.append(task)
        return task

    def remove_task(self, task):
        self._tasks.remove(task)

    def get_tasks(self):
        return self._tasks[:]

    def matches(self, path):
        """Return true if *all* tasks match the given path."""
        for task in self._tasks:
            if not task.matches(path):
                return False
        return bool(self._tasks)

    def run(self, path):
        """Run all tasks with the given action.

        Running an event means running all of its tasks individually and in
        order.  An event should only ever be run if all of its tasks claim to
        match the given action.

        The result of this method will be the last result of a task
        which isn't None, or None if they're all None.
        """
        result = None
        errors = []
        for task in self._tasks:
            try:
                task_result = task.run(path)
            except AssertionError, e:
                error = str(e)
                if not error:
                    raise RuntimeError("Empty error message from %r" % task)
                errors.append(error)
            else:
                if task_result is not None:
                    result = task_result
        if errors:
            message = [str(self.path)]
            for error in errors:
                lines = error.splitlines()
                message.append("- " + lines.pop(0))
                message.extend("  " + line for line in lines)
            raise AssertionError(os.linesep.join(message))
        return result

    def satisfied(self):
        """Return true if all tasks are satisfied.

        Being satisfied means that there are no unmet expectations.
        """
        for task in self._tasks:
            try:
                task.verify()
            except AssertionError:
                return False
        return True

    def verify(self):
        """Run verify on all tasks.

        The verify method is supposed to raise an AssertionError if the
        task has unmet expectations, with a one-line explanation about
        why this item is unmet.  This method should be safe to be called
        multiple times without side effects.
        """
        errors = []
        for task in self._tasks:
            try:
                task.verify()
            except AssertionError, e:
                error = str(e)
                if not error:
                    raise RuntimeError("Empty error message from %r" % task)
                errors.append(error)
        if errors:
            message = [str(self.path)]
            for error in errors:
                lines = error.splitlines()
                message.append("- " + lines.pop(0))
                message.extend("  " + line for line in lines)
            raise AssertionError(os.linesep.join(message))

    def set_state(self, state):
        """Change the task state of all tasks to reflect that of the mocker.

        State is either REPLAY, RECORD, or RESTORE.
        """
        for task in self._tasks:
            task.set_state(state)


class Task(object):
    """Minor item used for composition of a major task.

    A task item is responsible for adding any kind of logic to a
    task.  Examples of that are counting the number of times the
    task was made, verifying parameters if any, and so on.
    """

    def matches(self, path):
        """Return true if the task is supposed to be run for the given path.
        """
        return True

    def run(self, path):
        """Perform the task item, considering that the given action happened.
        """

    def verify(self):
        """Raise AssertionError if expectations for this item are unmet.

        The verify method is supposed to raise an AssertionError if the
        task has unmet expectations, with a one-line explanation about
        why this item is unmet.  This method should be safe to be called
        multiple times without side effects.
        """

    def set_state(self, state):
        """Perform actions needed to reflect state of the mocker.

        State is either REPLAY, RECORD, or RESTORE.
        """

# --------------------------------------------------------------------
# Task implementations.

class PathMatcher(Task):
    """Match the action path against a given path."""

    def __init__(self, path):
        self.path = path

    def matches(self, path):
        return self.path.matches(path)

@recorder
def path_matcher_recorder(mocker, event):
    event.add_task(PathMatcher(event.path))


class RunCounter(Task):
    """Task which verifies if the number of runs are within given boundaries.
    """

    def __init__(self, min, max=False):
        self.min = min
        if max is None:
            self.max = sys.maxint
        elif max is False:
            self.max = min
        else:
            self.max = max
        self._runs = 0

    def run(self, path):
        self._runs += 1
        if self._runs > self.max:
            self.verify()

    def verify(self):
        if not self.min <= self._runs <= self.max:
            if self.max == sys.maxint:
                raise AssertionError("Expected at least %d time(s), "
                                     "seen %d time(s)."
                                     % (self.min, self._runs))
            if self.min == self.max:
                raise AssertionError("Expected %d time(s), seen %d time(s)."
                                     % (self.min, self._runs))
            raise AssertionError("Expected %d to %d time(s), seen %d time(s)."
                                 % (self.min, self.max, self._runs))

class ImplicitRunCounter(RunCounter):
    """RunCounter inserted by default on any event.

    This is a way to differentiate explicitly added counters and
    implicit ones.
    """

@recorder
def run_counter_recorder(mocker, event):
    """In principle, any event may be repeated once."""
    event.add_task(ImplicitRunCounter(1))

@recorder
def run_counter_removal_recorder(mocker, event):
    """
    Events created by getattr actions which lead to other events
    may be repeated any number of times. For that, we remove implicit
    run counters of any getattr actions leading to current one.
    """
    parent_path = event.path.parent_path
    for event in reversed(mocker.get_events()):
        if (event.path is parent_path and
            event.path.actions[-1].kind == "getattr"):
            for task in event.get_tasks():
                if type(task) is ImplicitRunCounter:
                    event.remove_task(task)


class MockReturner(Task):
    """Return a mock based on the action path."""

    def __init__(self, mocker):
        self.mocker = mocker

    def run(self, path):
        return Mock(self.mocker, path)

@recorder
def mock_returner_recorder(mocker, event):
    """Events that lead to other events must return mock objects."""
    parent_path = event.path.parent_path
    for event in mocker.get_events():
        if event.path is parent_path:
            for task in event.get_tasks():
                if isinstance(task, MockReturner):
                    break
            else:
                event.add_task(MockReturner(mocker))
            break


class FunctionRunner(Task):
    """Task that runs a function everything it's run.

    Arguments of the last action in the path are passed to the function,
    and the function result is also returned.
    """

    def __init__(self, func):
        self._func = func

    def run(self, path):
        action = path.actions[-1]
        return self._func(*action.args, **action.kwargs)


class PathExecuter(Task):
    """Task that executes a path in the real object, and returns the result."""

    def run(self, path):
        return path.execute(path.root_object)


class Orderer(Task):
    """Task to establish an order relation between two events.

    An orderer task will only match once all its dependencies have
    been run.
    """

    def __init__(self):
        self._run = False 
        self._dependencies = []

    def run(self, path):
        self._run = True

    def has_run(self):
        return self._run

    def add_dependency(self, orderer):
        self._dependencies.append(orderer)

    def get_dependencies(self):
        return self._dependencies

    def matches(self, path):
        for dependency in self._dependencies:
            if not dependency.has_run():
                return False
        return True


class SpecChecker(Task):
    """Task to check if arguments of the last action conform to a real method.
    """

    def __init__(self, method):
        self._method = method
        if method:
            self._args, self._varargs, self._varkwargs, self._defaults = \
                inspect.getargspec(method)
            if self._defaults is None:
                self._defaults = ()
            if type(method) is type(self.run):
                self._args = self._args[1:]

    def get_method(self):
        return self._method

    def _raise(self, message):
        spec = inspect.formatargspec(self._args, self._varargs,
                                     self._varkwargs, self._defaults)
        raise AssertionError("Specification is %s%s: %s" %
                             (self._method.__name__, spec, message))

    def run(self, path):
        if not self._method:
            raise AssertionError("Method not found in real specification")
        action = path.actions[-1]
        obtained_len = len(action.args)
        obtained_kwargs = action.kwargs.copy()
        nodefaults_len = len(self._args) - len(self._defaults)
        for i, name in enumerate(self._args):
            if i < obtained_len and name in action.kwargs:
                self._raise("%r provided twice" % name)
            if (i >= obtained_len and i < nodefaults_len and
                name not in action.kwargs):
                self._raise("%r not provided" % name)
            obtained_kwargs.pop(name, None)
        if obtained_len > len(self._args) and not self._varargs:
            self._raise("too many args provided")
        if obtained_kwargs and not self._varkwargs:
            self._raise("unknown kwargs: %s" % ", ".join(obtained_kwargs))

@recorder
def spec_checker_recorder(mocker, event):
    cls = event.path.root_mock.__mocker_spec__
    actions = event.path.actions
    if (cls and len(actions) == 2 and
        actions[0].kind == "getattr" and actions[1].kind == "call"):
        method = getattr(cls, actions[0].args[0], None)
        event.add_task(SpecChecker(method))


class ProxyReplacer(Task):
    """Task which installs and deinstalls proxy mocks.

    This task will replace a real object by a mock in all dictionaries
    found in the running interpreter via the garbage collecting system.
    """

    def __init__(self, mock):
        self.mock = mock
        self.__mocker_replace__ = False

    def matches(self, path):
        return False

    def set_state(self, state):
        if state is REPLAY:
            global_replace(self.mock.__mocker_object__, self.mock)
        else:
            global_replace(self.mock, self.mock.__mocker_object__)


def global_replace(remove, install):
    """Replace object 'remove' with object 'install' on all dictionaries."""
    for referrer in gc.get_referrers(remove):
        if (type(referrer) is dict and
            referrer.get("__mocker_replace__", True)):
            for key, value in referrer.iteritems():
                if value is remove:
                    referrer[key] = install


class Undefined(object):

    def __repr__(self):
        return "Undefined"

Undefined = Undefined()


class Patcher(Task):

    def __init__(self):
        super(Patcher, self).__init__()
        self._monitored = {} # {kind: {id(object): object}}
        self._patched = {}

    def matches(self, path):
        return False

    def is_monitoring(self, obj, kind):
        monitored = self._monitored.get(kind)
        if monitored:
            if id(obj) in monitored:
                return True
            cls = type(obj)
            if issubclass(cls, type):
                cls = obj
            bases = set(id(base) for base in cls.__mro__)
            bases.intersection_update(monitored)
            return bool(bases)
        return False

    def monitor(self, obj, kind):
        if kind not in self._monitored:
            self._monitored[kind] = {}
        self._monitored[kind][id(obj)] = obj

    def patch_attr(self, obj, attr, value):
        original = obj.__dict__.get(attr, Undefined)
        self._patched[id(obj), attr] = obj, attr, original
        setattr(obj, attr, value)

    def get_unpatched_attr(self, obj, attr):
        cls = type(obj)
        if issubclass(cls, type):
            cls = obj
        result = Undefined
        for mro_cls in cls.__mro__:
            key = (id(mro_cls), attr)
            if key in self._patched:
                result = self._patched[key][2]
                if result is not Undefined:
                    break
            elif attr in mro_cls.__dict__:
                result = mro_cls.__dict__.get(attr, Undefined)
                break
        if isinstance(result, object) and hasattr(type(result), "__get__"):
            if cls is obj:
                obj = None
            return result.__get__(obj, cls)
        return result

    def _get_kind_attr(self, kind):
        if kind == "getattr":
            return "__getattribute__"
        return "__%s__" % kind

    def set_state(self, state):
        if state is REPLAY:
            for kind in self._monitored:
                attr = self._get_kind_attr(kind)
                seen = set()
                for obj in self._monitored[kind].itervalues():
                    cls = type(obj)
                    if issubclass(cls, type):
                        cls = obj
                    if cls not in seen:
                        seen.add(cls)
                        unpatched = getattr(cls, attr, Undefined)
                        self.patch_attr(cls, attr,
                                        PatchedMethod(kind, unpatched,
                                                      self.is_monitoring))
                        self.patch_attr(cls, "__mocker_execute__",
                                        self.execute)
        else:
            for obj, attr, original in self._patched.itervalues():
                if original is Undefined:
                    delattr(obj, attr)
                else:
                    setattr(obj, attr, original)
            self._patched.clear()

    def execute(self, action, object):
        attr = self._get_kind_attr(action.kind)
        unpatched = self.get_unpatched_attr(object, attr)
        return unpatched(*action.args, **action.kwargs)


class PatchedMethod(object):

    def __init__(self, kind, unpatched, is_monitoring):
        self._kind = kind
        self._unpatched = unpatched
        self._is_monitoring = is_monitoring

    def __get__(self, obj, cls=None):
        object = obj or cls
        if not self._is_monitoring(object, self._kind):
            return self._unpatched.__get__(obj, cls)
        def method(*args, **kwargs):
            if self._kind == "getattr" and args[0].startswith("__mocker_"):
                return self._unpatched.__get__(obj, cls)(args[0])
            mock = object.__mocker_mock__
            return mock.__mocker_act__(self._kind, args, kwargs, object)
        return method


@recorder
def patcher_recorder(mocker, event):
    mock = event.path.root_mock
    if mock.__mocker_patcher__ and len(event.path.actions) == 1:
        patcher = mock.__mocker_patcher__
        patcher.monitor(mock.__mocker_object__, event.path.actions[0].kind)
