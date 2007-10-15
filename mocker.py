import inspect
import sys
import gc


__all__ = ["Mocker", "expect", "SAME", "CONTAINS", "ANY", "VARIOUS"]


# --------------------------------------------------------------------
# Exceptions

class UnexpectedExprError(AssertionError):
    """Raised when an unknown expression is seen in playback mode."""


# --------------------------------------------------------------------
# Helper for chained-style calling.

class expect(object):
    """This is a simple helper that allows a different call-style.

    With this class one can comfortably do chaining of calls to the
    mocker object responsible by the object being handler. For instance:

        expect(obj.attr).result(3).count(1, 2)

    Is the same as:

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
    """This is the implementation of Mocker, without any recorders."""

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

    def get_state(self):
        return self._state

    def _set_state(self, state):
        if state is not self._state:
            self._state = state
            for event in self._events:
                event.set_state(state)

    def replay(self):
        self._set_state(REPLAY)

    def record(self):
        self._set_state(RECORD)

    def restore(self):
        self._set_state(RESTORE)

    def is_ordering(self):
        return self._ordering

    def ordered(self):
        self._ordering = True
        return OrderedContext(self)

    def unordered(self):
        self._ordering = False
        self._last_orderer = None

    def get_events(self):
        return self._events[:]

    def add_event(self, event):
        self._events.append(event)
        if self._ordering:
            orderer = event.add_task(Orderer())
            if self._last_orderer:
                orderer.add_dependency(self._last_orderer)
            self._last_orderer = orderer
        return event

    def verify(self):
        for event in self._events:
            event.verify()

    def obj(self, spec=None):
        """Return a new mock object."""
        return Mock(self, spec=spec)

    def proxy(self, object, spec=None, name=None,
              passthrough=True, install=False):
        """Return a new mock object which proxies to the given object.
 
        Unknown expressions are passed through to the proxied object
        by default, unless passthrough is set to False.  When set to
        False, pass through only happens when explicitly requested.
        """
        mock = Mock(self, spec=spec, object=object,
                    name=name, passthrough=passthrough)
        if install:
            event = self.add_event(Event())
            event.add_task(ProxyInstaller(mock))
        return mock

    def module(self, name, passthrough=True):
        return self.proxy(__import__(name, {}, {}, [""]),
                          name=name, passthrough=passthrough, install=True)

    def act(self, path):
        """This is called by mock objects whenever something happens to them.
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
            raise UnexpectedExprError("[Mocker] Unexpected expression: %s"
                                      % path)
            # XXX Show the full state here.
        else:
            raise RuntimeError("Mocker isn't recording or replaying.")

    @classinstancemethod
    def get_recorders(cls, self):
        return (self or cls)._recorders[:]

    @classinstancemethod
    def add_recorder(cls, self, recorder):
        (self or cls)._recorders.append(recorder)
        return recorder

    @classinstancemethod
    def remove_recorder(cls, self, recorder):
        (self or cls)._recorders.remove(recorder)

    def result(self, value):
        """Last recorded event will return the given value."""
        self.call(lambda *args, **kwargs: value)

    def throw(self, exception):
        """Last recorded event will raise the given exception."""
        def raise_exception(*args, **kwargs):
            raise exception
        self.call(raise_exception)

    def call(self, func):
        """Last recorded event will cause the given function to be called.

        The result of the function will be used as the event result.
        """
        self._events[-1].add_task(FunctionRunner(func))

    def count(self, min, max=False):
        """Last recorded event must happen between min and max times.

        If the max argument isn't given, it's assumed to be the same
        as min.  If max is None, the event must just happen at least
        min times.
        """
        event = self._events[-1]
        for task in event.get_tasks():
            if isinstance(task, RunCounter):
                event.remove_task(task)
        event.add_task(RunCounter(min, max))

    def order(self, *path_holders):
        """Ensure that events referred to by given objects happen in order.

        Arguments passed to this method might be mock objects returned
        from recorded operations.
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

        Arguments passed to this method might be mock objects returned
        from recorded operations.
        """
        last_path = self._events[-1].path
        for path_holder in path_holders:
            self.order(path_holder, last_path)

    def before(self, *path_holders):
        """Last recorded event must happen before events referred to.

        Arguments passed to this method might be mock objects returned
        from recorded operations.
        """
        last_path = self._events[-1].path
        for path_holder in path_holders:
            self.order(last_path, path_holder)

    def nospec(self):
        """Don't check method specification of real class on last event."""
        event = self._events[-1]
        for task in event.get_tasks():
            if isinstance(task, SpecChecker):
                event.remove_task(task)

    def passthrough(self):
        event = self._events[-1]
        if not event.path.root_mock.__mocker_object__:
            raise TypeError("Mock object isn't a proxy")
        event.add_task(PathApplier())


class OrderedContext(object):

    def __init__(self, mocker):
        self._mocker = mocker

    def __enter__(self):
        return None

    def __exit__(self, type, value, traceback):
        self._mocker.unordered()


class Mocker(MockerBase):
    pass

# Decorator to add recorders on the standard Mocker class.
recorder = Mocker.add_recorder


# --------------------------------------------------------------------
# Mock object.

class Mock(object):

    def __init__(self, mocker, path=None, name=None, spec=None,
                 object=None, passthrough=False):
        self.__mocker__ = mocker
        self.__mocker_path__ = path or Path(self)
        self.__mocker_name__ = name
        self.__mocker_spec__ = spec
        self.__mocker_object__ = object
        self.__mocker_passthrough__ = passthrough

    def __mocker_act__(self, kind, *args, **kwargs):
        if self.__mocker_name__ is None:
            self.__mocker_name__ = find_object_name(self, 2)
        action = Action(self.__mocker_path__, kind, args, kwargs)
        path = self.__mocker_path__ + action
        try:
            return self.__mocker__.act(path)
        except UnexpectedExprError:
            root_mock = path.root_mock
            if (root_mock.__mocker_passthrough__ and 
                root_mock.__mocker_object__ is not None):
                return path.apply(root_mock.__mocker_object__)
            raise

    def __getattribute__(self, name):
        if name.startswith("__mocker_"):
            return super(Mock, self).__getattribute__(name)
        return self.__mocker_act__("getattr", name)

    def __call__(self, *args, **kwargs):
        return self.__mocker_act__("call", *args, **kwargs)


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

    def __init__(self, path, kind, args, kwargs):
        self.path = path
        self.kind = kind
        self.args = args
        self.kwargs = kwargs
        self._apply_cache = {}

    def apply(self, object):
        # This caching scheme may fail if the object gets deallocated before
        # the action, as the id might get reused.  It's somewhat easy to fix
        # that with a weakref callback.  For our uses, though, the object
        # should never get deallocated before the action itself, so we'll
        # just keep it simple.
        if id(object) in self._apply_cache:
            return self._apply_cache[id(object)]
        kind = self.kind
        if kind == "getattr":
            kind = "getattribute"
        method = getattr(object, "__%s__" % kind)
        result = method(*self.args, **self.kwargs)
        self._apply_cache[id(object)] = result
        return result


class Path(object):

    def __init__(self, root_mock, actions=()):
        self.root_mock = root_mock
        self.actions = tuple(actions)

    @property
    def parent_path(self):
        if not self.actions:
            return None
        return self.actions[-1].path

    def __add__(self, action):
        """Return a new path which includes the given action at the end."""
        return self.__class__(self.root_mock, self.actions + (action,))

    def __eq__(self, other):
        """Verify if the two paths are equal.
        
        Two paths are equal if they refer to the same mock object, and
        have the actions with equal kind, args and kwargs.
        """
        if (self.root_mock is not other.root_mock or
            len(self.actions) != len(other.actions)):
            return False
        for action, other_action in zip(self.actions, other.actions):
            if (action.kind != other_action.kind or
                action.args != other_action.args or
                action.kwargs != other_action.kwargs):
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
            if (action.kind != other_action.kind or
                not match_params(action.args, action.kwargs,
                                 other_action.args, other_action.kwargs)):
                return False
        return True

    def apply(self, object):
        """Apply all actions sequentially on object, and return result.
        """
        for action in self.actions:
            object = action.apply(object)
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
    """Markup base for special arguments for matching parameters."""


class ANY(SpecialArgument):
    def __eq__(self, other):
        return not isinstance(other, SpecialArgument) or self is other
    def __repr__(self):
        return "ANY"
ANY = ANY()


class VARIOUS(SpecialArgument):
    def __repr__(self):
        return "VARIOUS"
    def __eq__(self, other):
        return self is other
VARIOUS = VARIOUS()


class SAME(SpecialArgument):
    def __init__(self, object):
        self.object = object
    def __repr__(self):
        return "SAME(%r)" % (self.object,)
    def __eq__(self, other):
        if isinstance(other, SpecialArgument):
            return type(other) == type(self) and self.object is other.object
        return self.object is other


class CONTAINS(SpecialArgument):
    def __init__(self, object):
        self.object = object
    def __repr__(self):
        return "CONTAINS(%r)" % (self.object,)
    def __eq__(self, other):
        if isinstance(other, SpecialArgument):
            return type(other) == type(self) and self.object == other.object
        return self.object in other


def match_params(args1, kwargs1, args2, kwargs2):
    """Match the two sets of parameters, considering the special VARIOUS."""

    # If they are equal, we're done.
    if args1 == args2 and kwargs1 == kwargs2:
        return True

    # Then, only if we have a VARIOUS argument we have a chance of matching.
    if VARIOUS not in args1:
        return False

    # Any keyword requests should be honored, but unrequested keywords
    # are also accepted, since the user is fine with whatever (VARIOUS!).
    for key, value in kwargs1.iteritems():
        if kwargs2.get(key) != value:
            return False

    # Easy choice. Keywords are matching, and anything on args are accepted.
    if (VARIOUS,) == args1:
        return True

    # We have something different there. If we don't have positional
    # arguments on the original call, it can't match.
    if not args2:
        # Unless we have just several VARIOUS (which is bizarre, but..).
        for arg1 in args1:
            if arg1 is not VARIOUS:
                return False
        return True

    # Ok, all bets are lost.  We have to actually do the more expensive
    # matching.  This is an algorithm based on the idea of the Levenshtein
    # Distance between two strings, but heavily hacked for this purpose.
    args2l = len(args2)
    if args1[0] is VARIOUS:
        args1 = args1[1:]
        array = [0]*args2l
    else:
        array = [1]*args2l
    for i in range(len(args1)):
        last = array[0]
        if args1[i] is VARIOUS:
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
        for task in self._tasks:
            task_result = task.run(path)
            if task_result is not None:
                result = task_result
        return result

    def satisfied(self):
        """Return true if all tasks are satisfied.

        Being satisfied means that there are no unmet expectations.
        """
        try:
            self.verify()
        except AssertionError:
            return False
        return True

    def verify(self):
        """Run verify on all tasks.

        The verify method is supposed to raise an AssertionError if the
        task has unmet expectations, with a nice debugging message
        explaining why it wasn't met.
        """
        for task in self._tasks:
            task.verify()

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

        The exception should include a nice explanation about why this
        item is unmet.
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


class PathApplier(Task):
    """Task that applies a path in the real object, and returns the result."""

    def run(self, path):
        return path.apply(path.root_mock.__mocker_object__)


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

    def run(self, path):
        if not self._method:
            raise AssertionError("method not existent in real class")
        action = path.actions[-1]
        obtained_len = len(action.args)
        obtained_kwargs = action.kwargs.copy()
        nodefaults_len = len(self._args) - len(self._defaults)
        for i, name in enumerate(self._args):
            if i < obtained_len and name in action.kwargs:
                raise AssertionError("%r parameter provided twice" % name)
            if (i >= obtained_len and i < nodefaults_len and
                name not in action.kwargs):
                raise AssertionError("%r parameter not provided" % name)
            obtained_kwargs.pop(name, None)
        if obtained_len > len(self._args) and not self._varargs:
            raise AssertionError("maximum number of parameters exceeded")
        if obtained_kwargs and not self._varkwargs:
            raise AssertionError("unknown kwargs: %s" %
                                 ", ".join(obtained_kwargs))

@recorder
def spec_checker_recorder(mocker, event):
    cls = event.path.root_mock.__mocker_spec__
    actions = event.path.actions
    if (cls and len(actions) == 2 and
        actions[0].kind == "getattr" and actions[1].kind == "call"):
        method = getattr(cls, actions[0].args[0], None)
        event.add_task(SpecChecker(method))


class ProxyInstaller(Task):
    """Task which installs and deinstalls proxy mocks.

    This task will replace a real object by a mock in all dictionaries
    found in the running interpreter via the garbage collecting system.
    """

    def __init__(self, mock):
        self.mock = mock

    def matches(self, path):
        return False

    def set_state(self, state):
        if state is REPLAY:
            install, remove = self.mock, self.mock.__mocker_object__
        else:
            install, remove = self.mock.__mocker_object__, self.mock
        mock_dict = object.__getattribute__(self.mock, "__dict__")
        protected = set((id(self.__dict__), id(mock_dict)))
        for referrer in gc.get_referrers(remove):
            if id(referrer) not in protected and type(referrer) is dict:
                for key, value in referrer.iteritems():
                    if value is remove:
                        referrer[key] = install
