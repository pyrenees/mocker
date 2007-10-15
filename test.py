#!/usr/bin/python
import unittest
import sys
import gc

from types import ModuleType

from mocker import (
    MockerBase, Mocker, Mock, Event, Task, Action, Path, recorder, expect,
    PathMatcher, path_matcher_recorder, RunCounter, ImplicitRunCounter,
    run_counter_recorder, run_counter_removal_recorder, MockReturner,
    mock_returner_recorder, FunctionRunner, Orderer, SpecChecker,
    spec_checker_recorder, match_params, ANY, VARIOUS, SAME, CONTAINS,
    UnexpectedExprError, PathApplier, RECORD, REPLAY, RESTORE, ProxyInstaller)


class CleanMocker(MockerBase):
    pass


class IntegrationTest(unittest.TestCase):

    def setUp(self):
        self.mocker = Mocker()

    def test_count(self):
        obj = self.mocker.obj()
        obj.x
        self.mocker.count(2, 3)

        self.mocker.replay()
        obj.x
        self.assertRaises(AssertionError, self.mocker.verify)
        obj.x
        self.mocker.verify()
        obj.x
        self.mocker.verify()
        self.assertRaises(AssertionError, getattr, obj, "x")

    def test_ordered(self):
        obj = self.mocker.obj()

        with_manager = self.mocker.ordered()
        with_manager.__enter__()
        obj.x
        obj.y
        obj.z
        with_manager.__exit__(None, None, None)

        self.mocker.replay()

        self.assertRaises(AssertionError, getattr, obj, "y")
        self.assertRaises(AssertionError, getattr, obj, "z")
        obj.x
        self.assertRaises(AssertionError, getattr, obj, "z")
        obj.y
        obj.z

    def test_spec(self):
        class C(object):
            def m(self, a): pass
        
        obj = self.mocker.obj(C)

        obj.m(1)
        obj.m(a=1)
        obj.m(1, 2)
        obj.m(b=2)
        obj.x()
        obj.y()
        self.mocker.nospec()
        obj.z()

        self.mocker.replay()

        obj.m(1)
        obj.m(a=1)
        obj.y()
        self.assertRaises(AssertionError, obj.m, 1, 2)
        self.assertRaises(AssertionError, obj.m, b=2)
        self.assertRaises(AssertionError, obj.x)
        self.assertRaises(AssertionError, obj.z)

    def test_result(self):
        obj = self.mocker.obj()
        obj.x
        self.mocker.result(42)
        self.mocker.replay()
        self.assertEquals(obj.x, 42)

    def test_throw(self):
        obj = self.mocker.obj()
        obj.x()
        self.mocker.throw(ValueError)
        self.mocker.replay()
        self.assertRaises(ValueError, obj.x)

    def test_call(self):
        calls = []
        def func(arg):
            calls.append(arg)
            return 42
        obj = self.mocker.obj()
        obj.x(24)
        self.mocker.call(func)
        self.mocker.replay()
        self.assertEquals(obj.x(24), 42)
        self.assertEquals(calls, [24])

    def test_call_result(self):
        calls = []
        def func(arg):
            calls.append(arg)
            return arg
        obj = self.mocker.obj()
        obj.x(24)
        self.mocker.call(func)
        self.mocker.result(42)
        self.mocker.replay()
        self.assertEquals(obj.x(24), 42)
        self.assertEquals(calls, [24])

    def test_proxy(self):
        class C(object):
            def sum(self, *args):
                return sum(args)
        
        obj = self.mocker.proxy(C())
        expect(obj.multiply(2, 3)).result(6)
        expect(obj.sum(0, 0)).result(1)
        expect(obj.sum(0, 0)).passthrough()

        self.mocker.replay()

        self.assertEquals(obj.multiply(2, 3), 6) # Mocked.
        self.assertRaises(AttributeError, obj.multiply) # Passed through.

        self.assertEquals(obj.sum(2, 3), 5) # Passed through.
        self.assertEquals(obj.sum(0, 0), 1) # Mocked.
        self.assertEquals(obj.sum(0, 0), 0) # Passed through explicitly.
        self.assertRaises(AssertionError, obj.sum, 0, 0) # Seen twice.

    def test_module_install_and_restore(self):
        try:
            module = self.mocker.module("calendar")
            import calendar
            self.assertTrue(calendar is not module)
            self.mocker.replay()
            import calendar
            self.assertTrue(calendar is module)
            self.mocker.restore()
            import calendar
            self.assertTrue(calendar is not module)
        finally:
            self.mocker.restore()

    def test_module_os_path(self):
        try:
            path = self.mocker.module("os.path")
            expect(path.join(VARIOUS)).call(lambda *args: "-".join(args))
            expect(path.join("e", VARIOUS)).passthrough()
            self.mocker.replay()
            import os
            self.assertEquals(os.path.join("a", "b", "c"), "a-b-c")
            self.assertNotEquals(os.path.join("e", "f", "g"), "e-f-g")
        finally:
            self.mocker.restore()


class ExpectTest(unittest.TestCase):

    def setUp(self):
        self.mocker = CleanMocker()

    def test_calling_mocker(self):
        obj = self.mocker.obj()
        expect(obj.attr).result(123)
        self.mocker.replay()
        self.assertEquals(obj.attr, 123)

    def test_chaining(self):
        obj = self.mocker.obj()
        expect(obj.attr).result(123).result(42)
        self.mocker.replay()
        self.assertEquals(obj.attr, 42)


class MockerTest(unittest.TestCase):

    def setUp(self):
        self.recorded = []
        self.mocker = CleanMocker()
        @self.mocker.add_recorder
        def recorder(mocker, event):
            self.recorded.append((mocker, event))

        self.action = Action(Path(Mock(self.mocker, name="mock")),
                             "getattr", ("attr",), {})
        self.path = self.action.path + self.action

    def test_default_state(self):
        self.assertEquals(self.mocker.get_state(), RECORD)

    def test_replay(self):
        calls = []
        event = self.mocker.add_event(Event())
        task = event.add_task(Task())
        task.set_state = lambda state: calls.append(state)
        self.mocker.replay()
        self.mocker.replay()
        self.assertEquals(self.mocker.get_state(), REPLAY)
        self.assertEquals(calls, [REPLAY])

    def test_record(self):
        calls = []
        event = self.mocker.add_event(Event())
        task = event.add_task(Task())
        task.set_state = lambda state: calls.append(state)
        self.mocker.replay()
        self.mocker.record()
        self.mocker.record()
        self.assertEquals(self.mocker.get_state(), RECORD)
        self.assertEquals(calls, [REPLAY, RECORD])

    def test_restore(self):
        calls = []
        event = self.mocker.add_event(Event())
        task = event.add_task(Task())
        task.set_state = lambda state: calls.append(state)
        self.mocker.restore()
        self.mocker.restore()
        self.assertEquals(self.mocker.get_state(), RESTORE)
        self.assertEquals(calls, [RESTORE])

    def test_verify(self):
        calls = []
        class MyEvent(object):
            def __init__(self, name):
                self.name = name
            def verify(self):
                calls.append(self.name)
        self.mocker.add_event(MyEvent("1"))
        self.mocker.add_event(MyEvent("2"))

        self.mocker.verify()

        self.assertEquals(calls, ["1", "2"])

    def test_add_recorder_on_instance(self):
        obj1 = object()
        obj2 = object()
        mocker = CleanMocker()
        self.assertEquals(mocker.add_recorder(obj1), obj1)
        self.assertEquals(mocker.add_recorder(obj2), obj2)
        self.assertEquals(mocker.get_recorders(), [obj1, obj2])
        mocker = CleanMocker()
        self.assertEquals(mocker.add_recorder(obj1), obj1)
        self.assertEquals(mocker.get_recorders(), [obj1])

    def test_add_recorder_on_class(self):
        class MyMocker(CleanMocker):
            pass
        obj1 = object()
        obj2 = object()
        self.assertEquals(MyMocker.add_recorder(obj1), obj1)
        self.assertEquals(MyMocker.add_recorder(obj2), obj2)
        mocker = MyMocker()
        self.assertEquals(mocker.get_recorders(), [obj1, obj2])
        mocker = MyMocker()
        self.assertEquals(mocker.get_recorders(), [obj1, obj2])

    def test_add_recorder_on_subclass(self):
        class MyMocker1(CleanMocker):
            pass
        obj1 = object()
        MyMocker1.add_recorder(obj1)
        class MyMocker2(MyMocker1):
            pass
        obj2 = object()
        MyMocker2.add_recorder(obj2)
        self.assertEquals(MyMocker1.get_recorders(), [obj1])
        self.assertEquals(MyMocker2.get_recorders(), [obj1, obj2])

    def test_remove_recorder_on_instance(self):
        obj1 = object()
        obj2 = object()
        obj3 = object()
        class MyMocker(CleanMocker):
            pass
        MyMocker.add_recorder(obj1)
        MyMocker.add_recorder(obj2)
        MyMocker.add_recorder(obj3)
        mocker = MyMocker()
        mocker.remove_recorder(obj2)
        self.assertEquals(mocker.get_recorders(), [obj1, obj3])
        self.assertEquals(MyMocker.get_recorders(), [obj1, obj2, obj3])

    def test_remove_recorder_on_class(self):
        class MyMocker(CleanMocker):
            pass
        obj1 = object()
        obj2 = object()
        self.assertEquals(MyMocker.add_recorder(obj1), obj1)
        self.assertEquals(MyMocker.add_recorder(obj2), obj2)
        MyMocker.remove_recorder(obj1)
        self.assertEquals(MyMocker.get_recorders(), [obj2])

    def test_obj(self):
        self.mocker = CleanMocker()
        obj = self.mocker.obj()
        self.assertEquals(type(obj), Mock)

    def test_obj_with_spec(self):
        class C(object): pass
        self.mocker = CleanMocker()
        obj = self.mocker.obj(C)
        self.assertEquals(obj.__mocker_spec__, C)

    def test_proxy(self):
        original = object()
        self.mocker = CleanMocker()
        obj = self.mocker.proxy(original)
        self.assertEquals(type(obj), Mock)
        self.assertEquals(obj.__mocker_object__, original)

    def test_proxy_with_spec(self):
        original = object()
        class C(object): pass
        self.mocker = CleanMocker()
        obj = self.mocker.proxy(original, C)
        self.assertEquals(obj.__mocker_object__, original)
        self.assertEquals(obj.__mocker_spec__, C)

    def test_proxy_with_passthrough_false(self):
        original = object()
        class C(object): pass
        self.mocker = CleanMocker()
        obj = self.mocker.proxy(original, C, passthrough=False)
        self.assertEquals(obj.__mocker_object__, original)
        self.assertEquals(obj.__mocker_spec__, C)
        self.assertEquals(obj.__mocker_passthrough__, False)

    def test_proxy_install(self):
        from os import path
        obj = object()
        proxy = self.mocker.proxy(obj, install=True)
        self.assertEquals(type(proxy), Mock)
        self.assertEquals(type(proxy.__mocker_object__), object)
        self.assertEquals(proxy.__mocker_object__, obj)
        (event,) = self.mocker.get_events()
        (task,) = event.get_tasks()
        self.assertEquals(type(task), ProxyInstaller)
        self.assertTrue(task.mock is proxy)
        self.assertTrue(task.mock.__mocker_object__ is obj)
        self.assertTrue(proxy is not obj)

    def test_module(self):
        from os import path
        module = self.mocker.module("os.path")
        self.assertEquals(type(module), Mock)
        self.assertEquals(type(module.__mocker_object__), ModuleType)
        self.assertEquals(module.__mocker_name__, "os.path")
        self.assertEquals(module.__mocker_object__, path)
        (event,) = self.mocker.get_events()
        (task,) = event.get_tasks()
        self.assertEquals(type(task), ProxyInstaller)
        self.assertTrue(task.mock is module)
        self.assertTrue(task.mock.__mocker_object__ is path)
        self.assertTrue(module is not path)

    def test_module_with_passthrough_false(self):
        module = self.mocker.module("calendar", passthrough=False)
        self.assertEquals(module.__mocker_passthrough__, False)

    def test_add_and_get_event(self):
        self.mocker.add_event(41)
        self.assertEquals(self.mocker.add_event(42), 42)
        self.assertEquals(self.mocker.get_events(), [41, 42])

    def test_recording(self):
        obj = self.mocker.obj()
        obj.attr()

        self.assertEquals(len(self.recorded), 2)

        action1 = Action(None, "getattr", ("attr",), {})
        action2 = Action(None, "call", (), {})

        mocker1, event1 = self.recorded[0]
        self.assertEquals(mocker1, self.mocker)
        self.assertEquals(type(event1), Event)
        self.assertTrue(event1.path.matches(Path(obj, [action1])))

        mocker2, event2 = self.recorded[1]
        self.assertEquals(mocker2, self.mocker)
        self.assertEquals(type(event2), Event)
        self.assertTrue(event2.path.matches(Path(obj, [action1, action2])))

        self.assertEquals(self.mocker.get_events(), [event1, event2])

    def test_recording_result_path(self):
        obj = self.mocker.obj()
        result = obj.attr()
        path = Path(obj, [Action(None, "getattr", ("attr",), {}),
                          Action(None, "call", (), {})])
        self.assertTrue(result.__mocker_path__.matches(path))

    def test_replaying_no_events(self):
        self.mocker.replay()
        try:
            self.mocker.act(self.path)
        except AssertionError, e:
            pass
        else:
            self.fail("AssertionError not raised")
        self.assertEquals(str(e), "[Mocker] Unexpected expression: mock.attr")

    def test_replaying_matching(self):
        calls = []
        class MyTask(Task):
            def matches(_, path):
                calls.append("matches")
                self.assertTrue(self.path.matches(path))
                return True
            def run(_, path):
                calls.append("run")
                self.assertTrue(self.path.matches(path))
                return "result"
        event = Event()
        event.add_task(MyTask())
        self.mocker.add_event(event)
        self.mocker.replay()
        self.assertEquals(self.mocker.act(self.path), "result")
        self.assertEquals(calls, ["matches", "run"])

    def test_replaying_none_matching(self):
        calls = []
        class MyTask(Task):
            def matches(_, path):
                self.assertTrue(self.path.matches(path))
                calls.append("matches")
                return False
        event = Event()
        event.add_task(MyTask())
        self.mocker.add_event(event)
        self.mocker.replay()
        self.assertRaises(AssertionError, self.mocker.act, self.path)
        self.assertEquals(calls, ["matches"])

    def test_replaying_not_satisfied_first(self):
        class MyTask1(Task):
            def run(self, path):
                return "result1"
        class MyTask2(Task):
            raised = False
            def verify(self):
                if not self.raised:
                    self.raised = True
                    raise AssertionError()
            def run(self, path):
                return "result2"
        event1 = self.mocker.add_event(Event())
        event1.add_task(MyTask1())
        event2 = self.mocker.add_event(Event())
        event2.add_task(MyTask2())
        event3 = self.mocker.add_event(Event())
        event3.add_task(MyTask1())
        self.mocker.replay()
        self.assertEquals(self.mocker.act(self.path), "result2")
        self.assertEquals(self.mocker.act(self.path), "result1")

    def test_recorder_decorator(self):
        result = recorder(42)
        try:
            self.assertEquals(result, 42)
            self.assertEquals(Mocker.get_recorders()[-1], 42)
            self.assertEquals(MockerBase.get_recorders(), [])
        finally:
            Mocker.remove_recorder(42)

    def test_result(self):
        event1 = self.mocker.add_event(Event())
        event2 = self.mocker.add_event(Event())
        self.mocker.result(123)
        self.assertEquals(event2.run(self.path), 123)

    def test_throw(self):
        class MyException(Exception): pass
        event1 = self.mocker.add_event(Event())
        event2 = self.mocker.add_event(Event())
        self.mocker.throw(MyException)
        self.assertRaises(MyException, event2.run, self.path)

    def test_call(self):
        event1 = self.mocker.add_event(Event())
        event2 = self.mocker.add_event(Event())
        self.mocker.call(lambda *args, **kwargs: 123)
        self.assertEquals(event2.run(self.path), 123)

    def test_count(self):
        event1 = self.mocker.add_event(Event())
        event2 = self.mocker.add_event(Event())
        event2.add_task(ImplicitRunCounter(1))
        self.mocker.count(2, 3)
        self.assertEquals(len(event1.get_tasks()), 0)
        (task,) = event2.get_tasks()
        self.assertEquals(type(task), RunCounter)
        self.assertEquals(task.min, 2)
        self.assertEquals(task.max, 3)
        self.mocker.count(4)
        self.assertEquals(len(event1.get_tasks()), 0)
        (task,) = event2.get_tasks()
        self.assertEquals(type(task), RunCounter)
        self.assertEquals(task.min, 4)
        self.assertEquals(task.max, 4)

    def test_order(self):
        mock1 = self.mocker.obj()
        mock2 = self.mocker.obj()
        mock3 = self.mocker.obj()
        mock4 = self.mocker.obj()
        result1 = mock1.attr1(1)
        result2 = mock2.attr2(2)
        result3 = mock3.attr3(3)
        result4 = mock4.attr4(4)

        # Try to spoil the logic which decides which task to reuse.
        other_task = Task()
        for event in self.mocker.get_events():
            event.add_task(other_task)

        self.mocker.order(result1, result2, result3)
        self.mocker.order(result1, result4)
        self.mocker.order(result2, result4)
        events = self.mocker.get_events()
        self.assertEquals(len(events), 8)

        self.assertEquals(events[0].get_tasks(), [other_task])
        other_task_, task1 = events[1].get_tasks()
        self.assertEquals(type(task1), Orderer)
        self.assertEquals(task1.get_dependencies(), [])
        self.assertEquals(other_task_, other_task)

        self.assertEquals(events[2].get_tasks(), [other_task])
        other_task_, task3 = events[3].get_tasks()
        self.assertEquals(type(task3), Orderer)
        self.assertEquals(task3.get_dependencies(), [task1])
        self.assertEquals(other_task_, other_task)

        self.assertEquals(events[4].get_tasks(), [other_task])
        other_task_, task5 = events[5].get_tasks()
        self.assertEquals(type(task5), Orderer)
        self.assertEquals(task5.get_dependencies(), [task3])
        self.assertEquals(other_task_, other_task)

        self.assertEquals(events[6].get_tasks(), [other_task])
        other_task_, task7 = events[7].get_tasks()
        self.assertEquals(type(task7), Orderer)
        self.assertEquals(task7.get_dependencies(), [task1, task3])
        self.assertEquals(other_task_, other_task)

    def test_after(self):
        mock1 = self.mocker.obj()
        mock2 = self.mocker.obj()
        mock3 = self.mocker.obj()
        result1 = mock1.attr1(1)
        result2 = mock2.attr2(2)
        result3 = mock3.attr3(3)

        # Try to spoil the logic which decides which task to reuse.
        other_task = Task()
        for event in self.mocker.get_events():
            event.add_task(other_task)

        self.mocker.after(result1, result2)

        events = self.mocker.get_events()
        self.assertEquals(len(events), 6)

        self.assertEquals(events[0].get_tasks(), [other_task])
        other_task_, task1 = events[1].get_tasks()
        self.assertEquals(type(task1), Orderer)
        self.assertEquals(task1.get_dependencies(), [])
        self.assertEquals(other_task_, other_task)

        self.assertEquals(events[2].get_tasks(), [other_task])
        other_task_, task3 = events[3].get_tasks()
        self.assertEquals(type(task3), Orderer)
        self.assertEquals(task3.get_dependencies(), [])
        self.assertEquals(other_task_, other_task)

        self.assertEquals(events[4].get_tasks(), [other_task])
        other_task_, task5 = events[5].get_tasks()
        self.assertEquals(type(task5), Orderer)
        self.assertEquals(task5.get_dependencies(), [task1, task3])
        self.assertEquals(other_task_, other_task)

    def test_before(self):
        mock1 = self.mocker.obj()
        mock2 = self.mocker.obj()
        mock3 = self.mocker.obj()
        result1 = mock1.attr1(1)
        result2 = mock2.attr2(2)
        result3 = mock3.attr3(3)

        # Try to spoil the logic which decides which task to reuse.
        other_task = Task()
        for event in self.mocker.get_events():
            event.add_task(other_task)

        self.mocker.before(result1, result2)

        events = self.mocker.get_events()
        self.assertEquals(len(events), 6)

        self.assertEquals(events[4].get_tasks(), [other_task])
        other_task_, task5 = events[5].get_tasks()
        self.assertEquals(type(task5), Orderer)
        self.assertEquals(task5.get_dependencies(), [])
        self.assertEquals(other_task_, other_task)

        self.assertEquals(events[0].get_tasks(), [other_task])
        other_task_, task1 = events[1].get_tasks()
        self.assertEquals(type(task1), Orderer)
        self.assertEquals(task1.get_dependencies(), [task5])
        self.assertEquals(other_task_, other_task)

        self.assertEquals(events[2].get_tasks(), [other_task])
        other_task_, task3 = events[3].get_tasks()
        self.assertEquals(type(task3), Orderer)
        self.assertEquals(task3.get_dependencies(), [task5])
        self.assertEquals(other_task_, other_task)

    def test_default_ordering(self):
        self.assertEquals(self.mocker.is_ordering(), False)

    def test_ordered(self):
        self.mocker.ordered()
        self.assertEquals(self.mocker.is_ordering(), True)

    def test_ordered_enter_exit(self):
        with_manager = self.mocker.ordered()
        self.assertEquals(self.mocker.is_ordering(), True)
        with_manager.__enter__()
        self.assertEquals(self.mocker.is_ordering(), True)
        with_manager.__exit__(None, None, None)
        self.assertEquals(self.mocker.is_ordering(), False)

    def test_unordered(self):
        self.mocker.ordered()
        self.mocker.unordered()
        self.assertEquals(self.mocker.is_ordering(), False)

    def test_ordered_events(self):
        mock = self.mocker.obj()

        # Ensure that the state is correctly reset between
        # different ordered blocks.
        self.mocker.ordered()

        mock.a

        self.mocker.unordered()
        self.mocker.ordered()

        mock.x.y.z

        events = self.mocker.get_events()

        (task1,) = events[1].get_tasks()
        (task2,) = events[2].get_tasks()
        (task3,) = events[3].get_tasks()

        self.assertEquals(type(task1), Orderer)
        self.assertEquals(type(task2), Orderer)
        self.assertEquals(type(task3), Orderer)

        self.assertEquals(task1.get_dependencies(), [])
        self.assertEquals(task2.get_dependencies(), [task1])
        self.assertEquals(task3.get_dependencies(), [task2])

    def test_nospec(self):
        event1 = self.mocker.add_event(Event())
        event2 = self.mocker.add_event(Event())
        task1 = event1.add_task(SpecChecker(None))
        task2 = event2.add_task(Task())
        task3 = event2.add_task(SpecChecker(None))
        task4 = event2.add_task(Task())
        self.mocker.nospec()
        self.assertEquals(event1.get_tasks(), [task1])
        self.assertEquals(event2.get_tasks(), [task2, task4])

    def test_passthrough(self):
        obj = self.mocker.proxy(object())
        event1 = self.mocker.add_event(Event(Path(obj)))
        event2 = self.mocker.add_event(Event(Path(obj)))
        self.mocker.passthrough()
        self.assertEquals(event1.get_tasks(), [])
        (task,) = event2.get_tasks()
        self.assertEquals(type(task), PathApplier)

    def test_passthrough_fails_on_unproxied(self):
        obj = self.mocker.obj()
        event1 = self.mocker.add_event(Event(Path(obj)))
        event2 = self.mocker.add_event(Event(Path(obj)))
        self.assertRaises(TypeError, self.mocker.passthrough)

    def test_on(self):
        obj = self.mocker.obj()
        self.mocker.on(obj.attr).result(123)
        self.mocker.replay()
        self.assertEquals(obj.attr, 123)


class ActionTest(unittest.TestCase):

    def setUp(self):
        self.mock = Mock(None, name="mock")

    def test_create(self):
        objects = [object() for i in range(4)]
        action = Action(*objects)
        self.assertEquals(action.path, objects[0])
        self.assertEquals(action.kind, objects[1])
        self.assertEquals(action.args, objects[2])
        self.assertEquals(action.kwargs, objects[3])

    def test_apply_getattr(self):
        class C(object):
            pass
        obj = C()
        obj.x = C()
        action = Action(None, "getattr", ("x",), {})
        self.assertEquals(action.apply(obj), obj.x)

    def test_apply_call(self):
        obj = lambda a, b: a+b
        action = Action(None, "call", (1,), {"b": 2})
        self.assertEquals(action.apply(obj), 3)

    def test_apply_caching(self):
        values = iter(range(10))
        obj = lambda: values.next()
        action = Action(None, "call", (), {})
        self.assertEquals(action.apply(obj), 0)
        self.assertEquals(action.apply(obj), 0)
        obj = lambda: values.next()
        self.assertEquals(action.apply(obj), 1)


class PathTest(unittest.TestCase):

    def setUp(self):
        class StubMocker(object):
            @staticmethod
            def act(path):
                pass
        self.mocker = StubMocker()
        self.mock = Mock(self.mocker, name="obj")

    def test_create(self):
        mock = object()
        path = Path(mock)
        self.assertEquals(path.root_mock, mock)
        self.assertEquals(path.actions, ())

    def test_create_with_actions(self):
        mock = object()
        path = Path(mock, [1,2,3])
        self.assertEquals(path.root_mock, mock)
        self.assertEquals(path.actions, (1,2,3))

    def test_add(self):
        mock = object()
        path = Path(mock, [1,2,3])
        result = path + 4
        self.assertTrue(result is not path)
        self.assertEquals(result.root_mock, mock)
        self.assertEquals(result.actions, (1,2,3,4))

    def test_parent_path(self):
        path1 = Path(self.mock)
        path2 = path1 + Action(path1, "getattr", ("attr",), {})
        path3 = path2 + Action(path2, "getattr", ("attr",), {})

        self.assertEquals(path1.parent_path, None)
        self.assertEquals(path2.parent_path, path1)
        self.assertEquals(path3.parent_path, path2)

    def test_equals(self):
        mock = object()
        obj1 = object()
        obj2 = object()

        # Not the *same* mock.
        path1 = Path([], [])
        path2 = Path([], [])
        self.assertNotEquals(path1, path2)

        path1 = Path(mock, [Action(obj1, "kind", (), {})])
        path2 = Path(mock, [Action(obj2, "kind", (), {})])
        self.assertEquals(path1, path2)

        path1 = Path(mock, [Action(obj1, "kind", (), {})])
        path2 = Path(mock, [Action(obj2, "dnik", (), {})])
        self.assertNotEquals(path1, path2)

        path1 = Path(mock, [Action(obj1, "kind", (), {})])
        path2 = Path(object(), [Action(obj2, "kind", (), {})])
        self.assertNotEquals(path1, path2)

        path1 = Path(mock, [Action(obj1, "kind", (), {})])
        path2 = Path(mock, [Action(obj2, "kind", (1,), {})])
        self.assertNotEquals(path1, path2)

        path1 = Path(mock, [Action(obj1, "kind", (), {})])
        path2 = Path(mock, [Action(obj2, "kind", (), {"a": 1})])
        self.assertNotEquals(path1, path2)

        path1 = Path(mock, [Action(obj1, "kind", (), {})])
        path2 = Path(mock, [])
        self.assertNotEquals(path1, path2)

        path1 = Path(mock, [Action(obj1, "kind", (ANY,), {})])
        path2 = Path(mock, [Action(obj2, "kind", (1), {})])
        self.assertNotEquals(path1, path2)

        path1 = Path(mock, [Action(obj1, "kind", (CONTAINS(1),), {})])
        path2 = Path(mock, [Action(obj2, "kind", (CONTAINS(1),), {})])
        self.assertEquals(path1, path2)

    def test_matches(self):
        mock = object()
        obj1 = object()
        obj2 = object()

        # Not the *same* mock.
        path1 = Path([], [])
        path2 = Path([], [])
        self.assertFalse(path1.matches(path2))

        path1 = Path(mock, [Action(obj1, "kind", (), {})])
        path2 = Path(mock, [Action(obj2, "kind", (), {})])
        self.assertTrue(path1.matches(path2))

        path1 = Path(mock, [Action(obj1, "kind", (), {})])
        path2 = Path(mock, [Action(obj2, "dnik", (), {})])
        self.assertFalse(path1.matches(path2))

        path1 = Path(mock, [Action(obj1, "kind", (), {})])
        path2 = Path(object(), [Action(obj2, "kind", (), {})])
        self.assertFalse(path1.matches(path2))

        path1 = Path(mock, [Action(obj1, "kind", (), {})])
        path2 = Path(mock, [Action(obj2, "kind", (1,), {})])
        self.assertFalse(path1.matches(path2))

        path1 = Path(mock, [Action(obj1, "kind", (), {})])
        path2 = Path(mock, [Action(obj2, "kind", (), {"a": 1})])
        self.assertFalse(path1.matches(path2))

        path1 = Path(mock, [Action(obj1, "kind", (), {})])
        path2 = Path(mock, [])
        self.assertFalse(path1.matches(path2))

        path1 = Path(mock, [Action(obj1, "kind", (VARIOUS,), {})])
        path2 = Path(mock, [Action(obj2, "kind", (), {})])
        self.assertTrue(path1.matches(path2))

        path1 = Path(mock, [Action(obj1, "kind", (VARIOUS,), {"a": 1})])
        path2 = Path(mock, [Action(obj2, "kind", (), {})])
        self.assertFalse(path1.matches(path2))

    def test_str(self):
        path = Path(self.mock, [])
        self.assertEquals(str(path), "obj")

    def test_str_unnamed(self):
        mock = Mock(self.mocker)
        path = Path(mock, [])
        self.assertEquals(str(path), "<mock>")

    def test_str_auto_named(self):
        named_mock = Mock(self.mocker)
        named_mock.attr
        path = Path(named_mock, [])
        self.assertEquals(str(path), "named_mock")

    def test_str_getattr(self):
        path = Path(self.mock, [Action(None, "getattr", ("attr",), {})])
        self.assertEquals(str(path), "obj.attr")

        path += Action(None, "getattr", ("x",), {})
        self.assertEquals(str(path), "obj.attr.x")

    def test_str_call(self):
        path = Path(self.mock, [Action(None, "call", (), {})])
        self.assertEquals(str(path), "obj()")

        path = Path(self.mock,
                    [Action(None, "call", (1, "2"), {"a":3,"b":"4"})])
        self.assertEquals(str(path), "obj(1, '2', a=3, b='4')")

    def test_str_getattr_call(self):
        path = Path(self.mock, [Action(None, "getattr", ("x",), {}),
                                Action(None, "getattr", ("y",), {}),
                                Action(None, "call", ("z",), {})])
        self.assertEquals(str(path), "obj.x.y('z')")

    def test_str_raise_on_unknown(self):
        path = Path(self.mock, [Action(None, "unknown", (), {})])
        self.assertRaises(RuntimeError, str, path)

    def test_apply(self):
        class C(object):
            pass
        obj = C()
        obj.x = C()
        obj.x.y = lambda a, b: a+b
        path = Path(self.mock, [Action(None, "getattr", ("x",), {}),
                                Action(None, "getattr", ("y",), {}),
                                Action(None, "call", (1,), {"b": 2})])
        self.assertEquals(path.apply(obj), 3)


class MatchParamsTest(unittest.TestCase):

    def true(self, *args):
        self.assertTrue(match_params(*args), repr(args))

    def false(self, *args):
        self.assertFalse(match_params(*args), repr(args))
    
    def test_any_repr(self):
        self.assertEquals(repr(ANY), "ANY")

    def test_any_equals(self):
        self.assertEquals(ANY, 1)
        self.assertEquals(ANY, 42)
        self.assertEquals(ANY, ANY)

    def test_various_repr(self):
        self.assertEquals(repr(VARIOUS), "VARIOUS")

    def test_various_equals(self):
        self.assertEquals(VARIOUS, VARIOUS)
        self.assertNotEquals(VARIOUS, ANY)
        self.assertNotEquals(ANY, VARIOUS)

    def test_same_repr(self):
        self.assertEquals(repr(SAME("obj")), "SAME('obj')")

    def test_same_equals(self):
        l1 = []
        l2 = []
        self.assertEquals(SAME(l1), l1)
        self.assertNotEquals(SAME(l1), l2)

        self.assertEquals(SAME(l1), SAME(l1))
        self.assertNotEquals(SAME(l1), SAME(l2))

        self.assertNotEquals(ANY, SAME(l1))
        self.assertNotEquals(SAME(l1), ANY)

    def test_contains_repr(self):
        self.assertEquals(repr(CONTAINS("obj")), "CONTAINS('obj')")

    def test_contains_equals(self):
        self.assertEquals(CONTAINS(1), [1])
        self.assertNotEquals(CONTAINS([1]), [1])

        self.assertEquals(CONTAINS([1]), CONTAINS([1]))
        self.assertNotEquals(CONTAINS(1), CONTAINS([1]))

        self.assertNotEquals(ANY, CONTAINS(1))
        self.assertNotEquals(CONTAINS(1), ANY)

    def test_normal(self):
        self.assertTrue(match_params((), {}, (), {}))
        self.assertTrue(match_params((1, 2), {"a": 3}, (1, 2), {"a": 3}))
        self.assertFalse(match_params((1,), {}, (), {}))
        self.assertFalse(match_params((), {}, (1,), {}))
        self.assertFalse(match_params((1, 2), {"a": 3}, (1, 2), {"a": 4}))
        self.assertFalse(match_params((1, 2), {"a": 3}, (1, 3), {"a": 3}))

    def test_any(self):
        self.assertTrue(match_params((1, 2), {"a": ANY}, (1, 2), {"a": 4}))
        self.assertTrue(match_params((1, ANY), {"a": 3}, (1, 3), {"a": 3}))
        self.assertFalse(match_params((ANY,), {}, (), {}))

    def test_various_alone(self):
        self.true((VARIOUS,), {}, (), {})
        self.true((VARIOUS,), {}, (1, 2), {})
        self.true((VARIOUS,), {}, (1, 2), {"a": 2})
        self.true((VARIOUS,), {}, (), {"a": 2})
        self.true((VARIOUS,), {"a": 1}, (), {"a": 1})
        self.true((VARIOUS,), {"a": 1}, (1, 2), {"a": 1})
        self.true((VARIOUS,), {"a": 1}, (), {"a": 1, "b": 2})
        self.true((VARIOUS,), {"a": 1}, (1, 2), {"a": 1, "b": 2})
        self.false((VARIOUS,), {"a": 1}, (), {})

    def test_various_at_start(self):
        self.true((VARIOUS, 3, 4), {}, (3, 4), {})
        self.true((VARIOUS, 3, 4), {}, (1, 2, 3, 4), {})
        self.true((VARIOUS, 3, 4), {"a": 1}, (3, 4), {"a": 1})
        self.true((VARIOUS, 3, 4), {"a": 1}, (1, 2, 3, 4), {"a": 1, "b": 2})
        self.false((VARIOUS, 3, 4), {}, (), {})
        self.false((VARIOUS, 3, 4), {}, (3, 5), {})
        self.false((VARIOUS, 3, 4), {}, (5, 5), {})
        self.false((VARIOUS, 3, 4), {"a": 1}, (), {})
        self.false((VARIOUS, 3, 4), {"a": 1}, (3, 4), {})
        self.false((VARIOUS, 3, 4), {"a": 1}, (3, 4), {"b": 2})

    def test_various_at_end(self):
        self.true((1, 2, VARIOUS), {}, (1, 2), {})
        self.true((1, 2, VARIOUS), {}, (1, 2, 3, 4), {})
        self.true((1, 2, VARIOUS), {"a": 1}, (1, 2), {"a": 1})
        self.true((1, 2, VARIOUS), {"a": 1}, (1, 2, 3, 4), {"a": 1, "b": 2})
        self.false((1, 2, VARIOUS), {}, (), {})
        self.false((1, 2, VARIOUS), {}, (1, 3), {})
        self.false((1, 2, VARIOUS), {}, (3, 3), {})
        self.false((1, 2, VARIOUS), {"a": 1}, (), {})
        self.false((1, 2, VARIOUS), {"a": 1}, (1, 2), {})
        self.false((1, 2, VARIOUS), {"a": 1}, (1, 2), {"b": 2})

    def test_various_at_middle(self):
        self.true((1, VARIOUS, 4), {}, (1, 4), {})
        self.true((1, VARIOUS, 4), {}, (1, 2, 3, 4), {})
        self.true((1, VARIOUS, 4), {"a": 1}, (1, 4), {"a": 1})
        self.true((1, VARIOUS, 4), {"a": 1}, (1, 2, 3, 4), {"a": 1, "b": 2})
        self.false((1, VARIOUS, 4), {}, (), {})
        self.false((1, VARIOUS, 4), {}, (1, 5), {})
        self.false((1, VARIOUS, 4), {}, (5, 5), {})
        self.false((1, VARIOUS, 4), {"a": 1}, (), {})
        self.false((1, VARIOUS, 4), {"a": 1}, (1, 4), {})
        self.false((1, VARIOUS, 4), {"a": 1}, (1, 4), {"b": 2})

    def test_various_multiple(self):
        self.true((VARIOUS, 3, VARIOUS, 6, VARIOUS), {},
                  (1, 2, 3, 4, 5, 6), {})
        self.true((VARIOUS, VARIOUS, VARIOUS), {}, (1, 2, 3, 4, 5, 6), {})
        self.true((VARIOUS, VARIOUS, VARIOUS), {},  (), {})
        self.false((VARIOUS, 3, VARIOUS, 6, VARIOUS), {},
                   (1, 2, 3, 4, 5), {})
        self.false((VARIOUS, 3, VARIOUS, 6, VARIOUS), {},
                   (1, 2, 4, 5, 6), {})


class MockTest(unittest.TestCase):

    def setUp(self):
        self.paths = []
        class StubMocker(object):
            @staticmethod
            def act(path):
                self.paths.append(path)
                return 42
        self.mocker = StubMocker()
        self.obj = Mock(self.mocker)

    def test_mocker(self):
        self.assertEquals(self.obj.__mocker__, self.mocker)
        self.assertEquals(self.obj.__mocker_name__, None)

    def test_default_path(self):
        path = self.obj.__mocker_path__
        self.assertEquals(path.root_mock, self.obj)
        self.assertEquals(path.actions, ())

    def test_path(self):
        path = object()
        self.assertEquals(Mock(self.mocker, path).__mocker_path__, path)

    def test_object(self):
        obj = Mock(self.mocker, object="foo")
        self.assertEquals(obj.__mocker_object__, "foo")

    def test_passthrough(self):
        obj = Mock(self.mocker, object="foo", passthrough=True)
        self.assertEquals(obj.__mocker_object__, "foo")
        self.assertEquals(obj.__mocker_passthrough__, True)

    def test_auto_naming(self):
        named_obj = self.obj
        named_obj.attr
        another_name = named_obj
        named_obj = None # Can't find this one anymore.
        another_name.attr
        self.assertEquals(another_name.__mocker_name__, "named_obj")

    def test_auto_naming_on_self(self):
        self.named_obj = self.obj
        del self.obj
        self.named_obj.attr
        self.assertEquals(self.named_obj.__mocker_name__, "named_obj")

    def test_auto_naming_on_bad_self(self):
        self_ = self
        self = object() # No __dict__
        self_.named_obj = self_.obj
        self_.named_obj.attr
        self_.assertEquals(self_.named_obj.__mocker_name__, None)

    def test_auto_naming_without_getframe(self):
        getframe = sys._getframe
        sys._getframe = None
        try:
            self.named_obj = self.obj
            self.named_obj.attr
            self.assertEquals(self.named_obj.__mocker_name__, None)
        finally:
            sys._getframe = getframe

    def test_getattr(self):
        self.assertEquals(self.obj.attr, 42)
        (path,) = self.paths
        self.assertEquals(type(path), Path)
        self.assertTrue(path.parent_path is self.obj.__mocker_path__)
        self.assertEquals(path, self.obj.__mocker_path__ + 
                                Action(None, "getattr", ("attr",), {}))

    def test_call(self):
        self.obj(1, a=2)
        (path,) = self.paths
        self.assertEquals(type(path), Path)
        self.assertTrue(path.parent_path is self.obj.__mocker_path__)
        self.assertEquals(path, self.obj.__mocker_path__ + 
                                Action(None, "call", (1,), {"a": 2}))

    def test_passthrough_on_unexpected(self):
        class StubMocker(object):
            def act(self, path):
                if path.actions[-1].args == ("x",):
                    raise UnexpectedExprError
                return 42
        class C(object):
            x = 123
            y = 321

        obj = Mock(StubMocker(), object=C())
        self.assertRaises(UnexpectedExprError, getattr, obj, "x", 42)
        self.assertEquals(obj.y, 42)

        obj = Mock(StubMocker(), object=C(), passthrough=True)
        self.assertEquals(obj.x, 123)
        self.assertEquals(obj.y, 42)


class EventTest(unittest.TestCase):

    def setUp(self):
        self.event = Event()

    def test_default_path(self):
        self.assertEquals(self.event.path, None)

    def test_path(self):
        path = object()
        event = Event(path)
        self.assertEquals(event.path, path)

    def test_add_and_get_tasks(self):
        task1 = self.event.add_task(Task())
        task2 = self.event.add_task(Task())
        self.assertEquals(self.event.get_tasks(), [task1, task2])

    def test_remove_task(self):
        task1 = self.event.add_task(Task())
        task2 = self.event.add_task(Task())
        task3 = self.event.add_task(Task())
        self.event.remove_task(task2)
        self.assertEquals(self.event.get_tasks(), [task1, task3])

    def test_default_matches(self):
        self.assertEquals(self.event.matches(None), False)

    def test_default_run(self):
        self.assertEquals(self.event.run(None), None)

    def test_default_satisfied(self):
        self.assertEquals(self.event.satisfied(), True)

    def test_default_verify(self):
        self.assertEquals(self.event.verify(), None)

    def test_default_restore(self):
        self.assertEquals(self.event.set_state(None), None)

    def test_matches_false(self):
        task1 = self.event.add_task(Task())
        task1.matches = lambda path: True
        task2 = self.event.add_task(Task())
        task2.matches = lambda path: False
        task3 = self.event.add_task(Task())
        task3.matches = lambda path: True
        self.assertEquals(self.event.matches(None), False)

    def test_matches_true(self):
        task1 = self.event.add_task(Task())
        task1.matches = lambda path: True
        task2 = self.event.add_task(Task())
        task2.matches = lambda path: True
        self.assertEquals(self.event.matches(None), True)

    def test_matches_argument(self):
        calls = []
        task = self.event.add_task(Task())
        task.matches = lambda path: calls.append(path)
        self.event.matches(42)
        self.assertEquals(calls, [42])

    def test_run(self):
        calls = []
        task1 = self.event.add_task(Task())
        task1.run = lambda path: calls.append(path) or True
        task2 = self.event.add_task(Task())
        task2.run = lambda path: calls.append(path) or False
        task3 = self.event.add_task(Task())
        task3.run = lambda path: calls.append(path) or None
        self.assertEquals(self.event.run(42), False)
        self.assertEquals(calls, [42, 42, 42])

    def test_satisfied_false(self):
        def raise_error():
            raise AssertionError
        task1 = self.event.add_task(Task())
        task2 = self.event.add_task(Task())
        task2.verify = raise_error
        task3 = self.event.add_task(Task())
        self.assertEquals(self.event.satisfied(), False)

    def test_satisfied_true(self):
        task1 = self.event.add_task(Task())
        task1.satisfied = lambda: True
        task2 = self.event.add_task(Task())
        task2.satisfied = lambda: True
        self.assertEquals(self.event.satisfied(), True)

    def test_verify(self):
        calls = []
        task1 = self.event.add_task(Task())
        task1.verify = lambda: calls.append(1)
        task2 = self.event.add_task(Task())
        task2.verify = lambda: calls.append(2)
        self.event.verify()
        self.assertEquals(calls, [1, 2])

    def test_set_state(self):
        calls = []
        task1 = self.event.add_task(Task())
        task1.set_state = lambda state: calls.append(state+1)
        task2 = self.event.add_task(Task())
        task2.set_state = lambda state: calls.append(state+2)
        self.event.set_state(3)
        self.assertEquals(calls, [4, 5])


class TaskTest(unittest.TestCase):

    def setUp(self):
        self.task = Task()

    def test_default_matches(self):
        self.assertEquals(self.task.matches(None), True)

    def test_default_run(self):
        self.assertEquals(self.task.run(None), None)

    def test_default_verify(self):
        self.assertEquals(self.task.verify(), None)

    def test_default_set_state(self):
        self.assertEquals(self.task.set_state(None), None)


class PathMatcherTest(unittest.TestCase):

    def setUp(self):
        self.mocker = CleanMocker()
        self.mock = self.mocker.obj()

    def test_is_task(self):
        self.assertTrue(isinstance(PathMatcher(None), Task))

    def test_create(self):
        path = object()
        task = PathMatcher(path)
        self.assertEquals(task.path, path)

    def test_matches(self):
        path = Path(self.mock, [Action(None, "getattr", ("attr1",), {})])
        task = PathMatcher(path)
        action = Action(Path(self.mock), "getattr", (), {})
        self.assertFalse(task.matches(action.path + action))
        action = Action(Path(self.mock), "getattr", ("attr1",), {})
        self.assertTrue(task.matches(action.path + action))

    def test_recorder(self):
        path = Path(self.mock, [Action(None, "call", (), {})])
        event = Event(path)
        path_matcher_recorder(self.mocker, event)
        (task,) = event.get_tasks()
        self.assertEquals(type(task), PathMatcher)
        self.assertTrue(task.path is path)

    def test_is_standard_recorder(self):
        self.assertTrue(path_matcher_recorder in Mocker.get_recorders())


class RunCounterTest(unittest.TestCase):

    def setUp(self):
        self.mocker = CleanMocker()
        self.mock = self.mocker.obj()
        self.action = Action(Path(self.mock), "getattr", ("attr",), {})
        self.path = Path(self.mock, [self.action])
        self.event = Event(self.path)

    def test_is_task(self):
        self.assertTrue(isinstance(RunCounter(1), Task))

    def test_create_one_argument(self):
        task = RunCounter(2)
        self.assertEquals(task.min, 2)
        self.assertEquals(task.max, 2)

    def test_create_min_max(self):
        task = RunCounter(2, 3)
        self.assertEquals(task.min, 2)
        self.assertEquals(task.max, 3)

    def test_create_unbounded(self):
        task = RunCounter(2, None)
        self.assertEquals(task.min, 2)
        self.assertEquals(task.max, sys.maxint)

    def test_run_one_argument(self):
        task = RunCounter(2)
        task.run(self.path)
        task.run(self.path)
        self.assertRaises(AssertionError, task.run, self.path)

    def test_run_two_arguments(self):
        task = RunCounter(1, 2)
        task.run(self.path)
        task.run(self.path)
        self.assertRaises(AssertionError, task.run, self.path)

    def test_verify(self):
        task = RunCounter(2)
        self.assertRaises(AssertionError, task.verify)
        task.run(self.path)
        self.assertRaises(AssertionError, task.verify)
        task.run(self.path)
        task.verify()
        self.assertRaises(AssertionError, task.run, self.path)
        self.assertRaises(AssertionError, task.verify)

    def test_verify_two_arguments(self):
        task = RunCounter(1, 2)
        self.assertRaises(AssertionError, task.verify)
        task.run(self.path)
        task.verify()
        task.run(self.path)
        task.verify()
        self.assertRaises(AssertionError, task.run, self.path)
        self.assertRaises(AssertionError, task.verify)

    def test_verify_unbound(self):
        task = RunCounter(1, None)
        self.assertRaises(AssertionError, task.verify)
        task.run(self.path)
        task.verify()
        task.run(self.path)
        task.verify()

    def test_recorder(self):
        run_counter_recorder(self.mocker, self.event)
        (task,) = self.event.get_tasks()
        self.assertEquals(type(task), ImplicitRunCounter)
        self.assertTrue(task.min == 1)
        self.assertTrue(task.max == 1)

    def test_removal_recorder(self):
        """
        Events created by getattr actions which lead to other events
        may be repeated any number of times.
        """
        path1 = Path(self.mock)
        path2 = path1 + Action(path1, "getattr", ("attr",), {})
        path3 = path2 + Action(path2, "getattr", ("attr",), {})
        path4 = path3 + Action(path3, "call", (), {})
        path5 = path4 + Action(path4, "call", (), {})

        event3 = self.mocker.add_event(Event(path3))
        event2 = self.mocker.add_event(Event(path2))
        event5 = self.mocker.add_event(Event(path5))
        event4 = self.mocker.add_event(Event(path4))

        event2.add_task(RunCounter(1))
        event2.add_task(ImplicitRunCounter(1))
        event2.add_task(RunCounter(1))
        event3.add_task(RunCounter(1))
        event3.add_task(ImplicitRunCounter(1))
        event3.add_task(RunCounter(1))
        event4.add_task(RunCounter(1))
        event4.add_task(ImplicitRunCounter(1))
        event4.add_task(RunCounter(1))
        event5.add_task(RunCounter(1))
        event5.add_task(ImplicitRunCounter(1))
        event5.add_task(RunCounter(1))
        
        # First, when the previous event isn't a getattr.

        run_counter_removal_recorder(self.mocker, event5)

        self.assertEquals(len(event2.get_tasks()), 3)
        self.assertEquals(len(event3.get_tasks()), 3)
        self.assertEquals(len(event4.get_tasks()), 3)
        self.assertEquals(len(event5.get_tasks()), 3)

        # Now, for real.

        run_counter_removal_recorder(self.mocker, event4)

        self.assertEquals(len(event2.get_tasks()), 3)
        self.assertEquals(len(event3.get_tasks()), 2)
        self.assertEquals(len(event4.get_tasks()), 3)
        self.assertEquals(len(event5.get_tasks()), 3)

        task1, task2 = event3.get_tasks()
        self.assertEquals(type(task1), RunCounter)
        self.assertEquals(type(task2), RunCounter)

    def test_removal_recorder_with_obj(self):

        self.mocker.add_recorder(run_counter_recorder)
        self.mocker.add_recorder(run_counter_removal_recorder)

        obj = self.mocker.obj()

        obj.x.y()()

        events = self.mocker.get_events()
        self.assertEquals(len(events), 4)
        self.assertEquals(len(events[0].get_tasks()), 0)
        self.assertEquals(len(events[1].get_tasks()), 0)
        self.assertEquals(len(events[2].get_tasks()), 1)
        self.assertEquals(len(events[3].get_tasks()), 1)

    def test_is_standard_recorder(self):
        self.assertTrue(run_counter_recorder in Mocker.get_recorders())
        self.assertTrue(run_counter_removal_recorder in Mocker.get_recorders())


class MockReturnerTest(unittest.TestCase):

    def setUp(self):
        self.mocker = CleanMocker()
        self.mock = self.mocker.obj()
        self.action = Action(Path(self.mock), "getattr", ("attr",), {})
        self.path = Path(self.mock, [self.action])
        self.event = Event(self.path)

    def test_is_task(self):
        self.assertTrue(isinstance(MockReturner(self.mocker), Task))

    def test_create(self):
        task = MockReturner(self.mocker)
        mock = task.run(self.path)
        self.assertTrue(isinstance(mock, Mock))
        self.assertEquals(mock.__mocker__, self.mocker)
        self.assertTrue(mock.__mocker_path__.matches(self.path))

    def test_recorder(self):
        path1 = Path(self.mock)
        path2 = path1 + Action(path1, "getattr", ("attr",), {})
        path3 = path2 + Action(path2, "getattr", ("attr",), {})
        path4 = path3 + Action(path3, "call", (), {})

        event2 = self.mocker.add_event(Event(path2))
        event3 = self.mocker.add_event(Event(path3))
        event4 = self.mocker.add_event(Event(path4))

        self.assertEquals(len(event2.get_tasks()), 0)
        self.assertEquals(len(event3.get_tasks()), 0)
        self.assertEquals(len(event4.get_tasks()), 0)

        # Calling on 4 should add it only to the parent.

        mock_returner_recorder(self.mocker, event4)

        self.assertEquals(len(event2.get_tasks()), 0)
        self.assertEquals(len(event3.get_tasks()), 1)
        self.assertEquals(len(event4.get_tasks()), 0)

        (task,) = event3.get_tasks()
        self.assertEquals(type(task), MockReturner)
        self.assertEquals(task.mocker, self.mocker)

        # Calling on it again shouldn't do anything.

        mock_returner_recorder(self.mocker, event4)

        self.assertEquals(len(event2.get_tasks()), 0)
        self.assertEquals(len(event3.get_tasks()), 1)
        self.assertEquals(len(event4.get_tasks()), 0)

    def test_is_standard_recorder(self):
        self.assertTrue(mock_returner_recorder in Mocker.get_recorders())


class FunctionRunnerTest(unittest.TestCase):

    def setUp(self):
        self.mocker = CleanMocker()
        self.mock = self.mocker.obj()
        self.action = Action(Path(self.mock), "call", (1, 2), {"c": 3})
        self.path = Path(self.mock, [self.action])
        self.event = Event(self.path)

    def test_is_task(self):
        self.assertTrue(isinstance(FunctionRunner(None), Task))

    def test_run(self):
        task = FunctionRunner(lambda *args, **kwargs: repr((args, kwargs)))
        result = task.run(self.path)
        self.assertEquals(result, "((1, 2), {'c': 3})")


class PathApplierTest(unittest.TestCase):

    def setUp(self):
        self.mocker = CleanMocker()

    def test_is_task(self):
        self.assertTrue(isinstance(PathApplier(), Task))

    def test_run(self):
        class C(object):
            pass
        obj = C()
        obj.x = C()
        obj.x.y = lambda a, b: a+b

        proxy = self.mocker.proxy(obj)

        path = Path(proxy, [Action(None, "getattr", ("x",), {}),
                            Action(None, "getattr", ("y",), {}),
                            Action(None, "call", (1,), {"b": 2})])

        task = PathApplier()
        self.assertEquals(task.run(path), 3)


class OrdererTest(unittest.TestCase):

    def setUp(self):
        self.mocker = CleanMocker()
        self.mock = self.mocker.obj()
        self.action = Action(Path(self.mock), "call", (1, 2), {"c": 3})
        self.path = Path(self.mock, [self.action])

    def test_is_task(self):
        self.assertTrue(isinstance(Orderer(), Task))

    def test_has_run(self):
        orderer = Orderer()
        self.assertFalse(orderer.has_run())
        orderer.run(self.path)
        self.assertTrue(orderer.has_run())

    def test_add_dependency_and_match(self):
        orderer1 = Orderer()
        orderer2 = Orderer()
        orderer2.add_dependency(orderer1)
        self.assertFalse(orderer2.matches(None))
        self.assertTrue(orderer1.matches(None))
        orderer1.run(self.path)
        self.assertTrue(orderer2.matches(None))

    def test_get_dependencies(self):
        orderer = Orderer()
        orderer.add_dependency(1)
        orderer.add_dependency(2)
        self.assertEquals(orderer.get_dependencies(), [1, 2])


class SpecCheckerTest(unittest.TestCase):

    def setUp(self):
        class C(object):
            def normal(self, a, b, c=3): pass
            def varargs(self, a, b, c=3, *args): pass
            def varkwargs(self, a, b, c=3, **kwargs): pass
            def varargskwargs(self, a, b, c=3, *args, **kwargs): pass
            @classmethod
            def klass(cls, a, b, c=3): pass
            @staticmethod
            def static(a, b, c=3): pass
            def noargs(self): pass
            @classmethod
            def klassnoargs(cls): pass
            @staticmethod
            def staticnoargs(): pass
        self.cls = C
        self.mocker = CleanMocker()
        self.mock = self.mocker.obj(self.cls)

    def path(self, *args, **kwargs):
        action = Action(Path(self.mock), "call", args, kwargs)
        return action.path + action

    def good(self, method_names, args_expr):
        if type(method_names) is not list:
            method_names = [method_names]
        for method_name in method_names:
            task = SpecChecker(getattr(self.cls, method_name, None))
            path = eval("self.path(%s)" % args_expr)
            try:
                task.run(path)
            except AssertionError:
                self.fail("AssertionError raised with self.cls.%s(%s)"
                          % (method_name, args_expr))

    def bad(self, method_names, args_expr):
        if type(method_names) is not list:
            method_names = [method_names]
        for method_name in method_names:
            task = SpecChecker(getattr(self.cls, method_name, None))
            path = eval("self.path(%s)" % args_expr)
            try:
                task.run(path)
            except AssertionError:
                pass
            else:
                self.fail("AssertionError not raised with self.cls.%s(%s)"
                          % (method_name, args_expr))

    def test_get_method(self):
        task = SpecChecker(self.cls.noargs)
        self.assertEquals(task.get_method(), self.cls.noargs)

    def test_is_standard_recorder(self):
        self.assertTrue(spec_checker_recorder in Mocker.get_recorders())

    def test_is_task(self):
        self.assertTrue(isinstance(SpecChecker(self.cls.normal), Task))

    def test_recorder(self):
        self.mocker.add_recorder(spec_checker_recorder)
        obj = self.mocker.obj(self.cls)
        obj.noargs()
        getattr, call = self.mocker.get_events()
        self.assertEquals(getattr.get_tasks(), [])
        (task,) = call.get_tasks()
        self.assertEquals(type(task), SpecChecker)
        self.assertEquals(task.get_method(), self.cls.noargs)

    def test_recorder_with_unexistent_method(self):
        self.mocker.add_recorder(spec_checker_recorder)
        obj = self.mocker.obj(self.cls)
        obj.unexistent()
        getattr, call = self.mocker.get_events()
        self.assertEquals(getattr.get_tasks(), [])
        (task,) = call.get_tasks()
        self.assertEquals(type(task), SpecChecker)
        self.assertEquals(task.get_method(), None)

    def test_recorder_second_action_isnt_call(self):
        self.mocker.add_recorder(spec_checker_recorder)
        obj = self.mocker.obj(self.cls)
        obj.noargs.x
        event1, event2 = self.mocker.get_events()
        self.assertEquals(event1.get_tasks(), [])
        self.assertEquals(event2.get_tasks(), [])

    def test_recorder_first_action_isnt_getattr(self):
        self.mocker.add_recorder(spec_checker_recorder)
        obj = self.mocker.obj(self.cls)
        obj("noargs").x
        event1, event2 = self.mocker.get_events()
        self.assertEquals(event1.get_tasks(), [])
        self.assertEquals(event2.get_tasks(), [])

    def test_recorder_more_than_two_actions(self):
        self.mocker.add_recorder(spec_checker_recorder)
        obj = self.mocker.obj(self.cls)
        obj.noargs().x
        event1, event2, event3 = self.mocker.get_events()
        self.assertEquals(len(event1.get_tasks()), 0)
        self.assertEquals(len(event2.get_tasks()), 1)
        self.assertEquals(len(event3.get_tasks()), 0)

    def test_noargs(self):
        methods = ["noargs", "klassnoargs", "staticnoargs"]
        self.good(methods, "")
        self.bad(methods, "1")
        self.bad(methods, "a=1")

    def test_args_and_kwargs(self):
        methods = ["normal", "varargs", "varkwargs", "varargskwargs",
                   "static", "klass"]
        self.good(methods, "1, 2")
        self.good(methods, "1, 2, 3")
        self.good(methods, "1, b=2")
        self.good(methods, "1, b=2, c=3")
        self.good(methods, "a=1, b=2")
        self.good(methods, "a=1, b=2, c=3")

    def test_too_much(self):
        methods = ["normal", "static", "klass"]
        self.bad(methods, "1, 2, 3, 4")
        self.bad(methods, "1, 2, d=4")

    def test_missing(self):
        methods = ["normal", "varargs", "varkwargs", "varargskwargs",
                   "static", "klass"]
        self.bad(methods, "")
        self.bad(methods, "1")
        self.bad(methods, "c=3")
        self.bad(methods, "a=1")
        self.bad(methods, "b=2, c=3")

    def test_duplicated_argument(self):
        methods = ["normal", "varargs", "varkwargs", "varargskwargs",
                   "static", "klass"]
        self.bad(methods, "1, 2, b=2")

    def test_varargs(self):
        self.good("varargs", "1, 2, 3, 4")
        self.bad("varargs", "1, 2, 3, 4, d=3")

    def test_varkwargs(self):
        self.good("varkwargs", "1, 2, d=3")
        self.bad("varkwargs", "1, 2, 3, 4, d=3")

    def test_varargskwargs(self):
        self.good("varargskwargs", "1, 2, 3, 4, d=3")

    def test_unexistent(self):
        self.bad("unexistent", "")


class ProxyInstallerTest(unittest.TestCase):

    def setUp(self):
        self.mocker = CleanMocker()
        import calendar
        self.mock = Mock(self.mocker, object=calendar)
        self.task = ProxyInstaller(self.mock)

    def tearDown(self):
        self.task.set_state(RESTORE)

    def test_is_task(self):
        self.assertTrue(isinstance(ProxyInstaller(None), Task))

    def test_mock(self):
        mock = object()
        task = ProxyInstaller(mock)
        self.assertEquals(task.mock, mock)

    def test_matches_nothing(self):
        self.assertFalse(self.task.matches(None))

    def test_defaults_to_not_installed(self):
        import calendar
        self.assertEquals(type(calendar), ModuleType)

    def test_install(self):
        self.task.set_state(REPLAY)
        import calendar
        self.assertEquals(type(calendar), Mock)
        self.assertTrue(calendar is self.mock)

    def test_install_protects_mock(self):
        self.task.set_state(REPLAY)
        self.assertEquals(type(self.mock.__mocker_object__), ModuleType)

    def test_deinstall_protects_task(self):
        self.task.set_state(REPLAY)
        self.task.set_state(RESTORE)
        self.assertEquals(type(self.task.mock), Mock)

    def test_install_on_object(self):
        class C(object):
            def __init__(self):
                import calendar
                self.calendar = calendar
        obj = C()
        self.task.set_state(REPLAY)
        self.assertEquals(type(obj.calendar), Mock)
        self.assertTrue(obj.calendar is self.mock)

    def test_install_on_submodule(self):
        from os import path
        mock = Mock(self.mocker, object=path)
        task = ProxyInstaller(mock)
        task.set_state(REPLAY)
        import os
        self.assertEquals(type(os.path), Mock)
        self.assertTrue(os.path is mock)

    def test_deinstall_on_restore(self):
        self.task.set_state(REPLAY)
        self.task.set_state(RESTORE)
        import calendar
        self.assertEquals(type(calendar), ModuleType)
        self.assertEquals(calendar.__name__, "calendar")

    def test_deinstall_on_record(self):
        self.task.set_state(REPLAY)
        self.task.set_state(RECORD)
        import calendar
        self.assertEquals(type(calendar), ModuleType)
        self.assertEquals(calendar.__name__, "calendar")

    def test_deinstall_from_object(self):
        class C(object):
            def __init__(self):
                import calendar
                self.calendar = calendar
        obj = C()
        self.task.set_state(REPLAY)
        self.task.set_state(RESTORE)
        self.assertEquals(type(obj.calendar), ModuleType)
        self.assertEquals(obj.calendar.__name__, "calendar")

    def test_deinstall_from_submodule(self):
        from os import path
        mock = Mock(self.mocker, object=path)
        task = ProxyInstaller(mock)
        task.set_state(REPLAY)
        task.set_state(RESTORE)
        import os
        self.assertEquals(type(os.path), ModuleType)


if __name__ == "__main__":
    unittest.main()
