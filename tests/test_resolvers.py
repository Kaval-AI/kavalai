from pydantic import BaseModel
from kavalai.resolvers import resolve_path, find_key_recursive


class SimpleModel(BaseModel):
    name: str
    age: int


class NestedModel(BaseModel):
    id: int
    simple: SimpleModel


class PlainObj:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_resolve_path_none_or_empty():
    obj = {"a": 1}
    assert resolve_path(obj, None) == obj
    assert resolve_path(obj, "") == obj


def test_resolve_path_dict():
    obj = {"a": {"b": 1}, "c": [1, 2, 3]}
    assert resolve_path(obj, "a.b") == 1
    assert resolve_path(obj, "c") == [1, 2, 3]
    assert resolve_path(obj, "d") is None


def test_resolve_path_list():
    obj = {"a": [1, 2, 3], "b": [{"id": 10}, {"id": 20}]}
    assert resolve_path(obj, "a.0") == 1
    assert resolve_path(obj, "a.2") == 3
    assert resolve_path(obj, "a.3") is None
    assert resolve_path(obj, "a.-1") is None
    assert resolve_path(obj, "a.not_a_digit") is None
    assert resolve_path(obj, "b.1.id") == 20
    # non-numeric access on list is not supported
    assert resolve_path([1, 2, 3], "not_a_digit") is None


def test_resolve_path_basemodel():
    model = NestedModel(id=1, simple=SimpleModel(name="test", age=30))
    assert resolve_path(model, "id") == 1
    assert resolve_path(model, "simple.name") == "test"
    assert resolve_path(model, "simple.age") == 30
    assert resolve_path(model, "invalid") is None


def test_resolve_path_plain_obj():
    obj = PlainObj(x=10, y=PlainObj(z=20))
    assert resolve_path(obj, "x") == 10
    assert resolve_path(obj, "y.z") == 20
    assert resolve_path(obj, "y.invalid") is None


def test_resolve_path_length():
    assert resolve_path([1, 2, 3], "length") == 3
    assert resolve_path({"a": 1, "b": 2}, "length") == 2
    assert resolve_path("string", "length") == 6
    assert resolve_path(123, "length") is None  # int has no __len__


def test_resolve_path_none_traversal():
    assert resolve_path({"a": None}, "a.b") is None
    assert resolve_path(None, "a") is None


def test_find_key_recursive_none():
    assert find_key_recursive(None, "target") is None


def test_find_key_recursive_dict():
    obj = {"a": 1, "b": {"c": 2}, "d": [{"e": 3}]}
    assert find_key_recursive(obj, "a") == 1
    assert find_key_recursive(obj, "c") == 2
    assert find_key_recursive(obj, "e") == 3
    assert find_key_recursive(obj, "f") is None


def test_find_key_recursive_basemodel():
    model = NestedModel(id=1, simple=SimpleModel(name="test", age=30))
    assert find_key_recursive(model, "id") == 1
    assert find_key_recursive(model, "name") == "test"
    assert find_key_recursive(model, "age") == 30
    assert find_key_recursive(model, "not_found") is None


def test_find_key_recursive_list():
    obj = [{"a": 1}, {"b": 2}]
    assert find_key_recursive(obj, "a") == 1
    assert find_key_recursive(obj, "b") == 2
    assert find_key_recursive(obj, "c") is None


def test_find_key_recursive_plain_obj():
    obj = PlainObj(x=10, child=PlainObj(y=20))
    assert find_key_recursive(obj, "x") == 10
    assert find_key_recursive(obj, "y") == 20
    assert find_key_recursive(obj, "z") is None


def test_resolve_path_tuple():
    obj = (1, 2, {"a": 3})
    assert resolve_path(obj, "0") == 1
    assert resolve_path(obj, "2.a") == 3
    assert resolve_path(obj, "3") is None


def test_resolve_path_basemodel_fallback():
    # Create a BaseModel that exposes a key only via model_dump (not as attribute)
    class DumpOnlyModel(BaseModel):
        x: int

        def model_dump(self, *args, **kwargs):  # type: ignore[override]
            data = super().model_dump(*args, **kwargs)
            data["dump_only"] = 42
            return data

    m = DumpOnlyModel(x=1)
    # Ensure attribute does not exist but dump contains it
    assert not hasattr(m, "dump_only")
    assert "dump_only" in m.model_dump()
    # This must go through the fallback branch using model_dump (lines 60-63)
    assert resolve_path(m, "dump_only") == 42


def test_find_key_recursive_basemodel_fallback():
    from pydantic import ConfigDict

    class ExtraModel(BaseModel):
        model_config = ConfigDict(extra="allow")

    model = ExtraModel(inner={"target": "found"})
    # target is inside a dict which is a value of an extra field
    # if it's not in model_fields, it might go to model_dump fallback
    assert find_key_recursive(model, "target") == "found"


def test_find_key_recursive_tuple():
    obj = ({"a": 1},)
    assert find_key_recursive(obj, "a") == 1


class UndumpableModel(BaseModel):
    """A model whose model_dump fails — the resolver must not propagate that."""

    name: str = "x"

    def model_dump(self, **kwargs):
        raise ValueError("cannot dump")


class FlakyAttribute:
    """hasattr() succeeds, the following getattr() fails."""

    def __init__(self):
        self.reads = 0

    @property
    def value(self):
        self.reads += 1
        if self.reads > 1:
            raise ValueError("boom")
        return 1


def test_resolve_path_returns_none_when_model_dump_fails():
    assert resolve_path(UndumpableModel(), "missing") is None


def test_resolve_path_returns_none_when_attribute_access_fails():
    assert resolve_path(FlakyAttribute(), "value") is None
