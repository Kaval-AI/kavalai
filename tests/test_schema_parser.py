import pytest
from pydantic import BaseModel

from kavalai.schema_parser import SchemaParser


# Assuming SchemaParser is imported from your module
# from my_parser import SchemaParser


def test_basic_parsing():
    """Test standard string and integer properties."""
    datatypes = {
        "User": {
            "type": "object",
            "properties": {"username": {"type": "string"}, "age": {"type": "integer"}},
        }
    }
    parser = SchemaParser(datatypes)
    models = parser.parse_all()

    User = models["User"]
    user_instance = User(username="alice", age=30)
    assert user_instance.username == "alice"
    assert user_instance.age == 30
    assert isinstance(user_instance, BaseModel)


def test_custom_ref_logic():
    """Test that $ref correctly points to other keys in the datatypes dict."""
    datatypes = {
        "Address": {"type": "object", "properties": {"city": {"type": "string"}}},
        "Profile": {"type": "object", "properties": {"location": {"$ref": "Address"}}},
    }
    parser = SchemaParser(datatypes)
    models = parser.parse_all()

    Profile = models["Profile"]
    Address = models["Address"]

    data = {"location": {"city": "New York"}}
    profile_instance = Profile(**data)

    assert isinstance(profile_instance.location, Address)
    assert profile_instance.location.city == "New York"


def test_array_parsing():
    """Test arrays of primitive types and arrays of references."""
    datatypes = {
        "Tag": {"type": "object", "properties": {"name": {"type": "string"}}},
        "Post": {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"$ref": "Tag"}},
                "scores": {"type": "array", "items": {"type": "integer"}},
            },
        },
    }
    parser = SchemaParser(datatypes)
    models = parser.parse_all()

    Post = models["Post"]
    Tag = models["Tag"]

    post_instance = Post(
        tags=[{"name": "python"}, {"name": "pydantic"}], scores=[10, 20, 30]
    )

    assert len(post_instance.tags) == 2
    assert isinstance(post_instance.tags[0], Tag)
    assert post_instance.scores == [10, 20, 30]


def test_missing_ref_raises_error():
    """Ensure the parser fails gracefully if a $ref is missing."""
    datatypes = {
        "Broken": {"type": "object", "properties": {"item": {"$ref": "NonExistent"}}}
    }
    parser = SchemaParser(datatypes)
    with pytest.raises(ValueError, match="Definition for 'NonExistent' not found"):
        parser.parse_all()


def test_optional_fields():
    """Test required: false makes a field optional."""
    datatypes = {
        "User": {
            "type": "object",
            "properties": {
                "username": {"type": "string"},
                "age": {"type": "integer", "required": False},
                "email": {"type": "string", "required": False, "default": "n/a"},
            },
        }
    }
    parser = SchemaParser(datatypes)
    models = parser.parse_all()

    User = models["User"]

    # age and email are optional
    user1 = User(username="alice")
    assert user1.username == "alice"
    assert user1.age is None
    assert user1.email == "n/a"

    # username is still required
    with pytest.raises(ValueError):
        User(age=30)


def test_required_default_is_true():
    """Test that fields are required by default."""
    datatypes = {
        "User": {
            "type": "object",
            "properties": {"username": {"type": "string"}},
        }
    }
    parser = SchemaParser(datatypes)
    models = parser.parse_all()
    User = models["User"]

    with pytest.raises(ValueError):
        User()


def test_optional_ref():
    """Test that a $ref can also be optional."""
    datatypes = {
        "Address": {"type": "object", "properties": {"city": {"type": "string"}}},
        "User": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {"$ref": "Address", "required": False},
            },
        },
    }
    parser = SchemaParser(datatypes)
    models = parser.parse_all()
    User = models["User"]

    user = User(name="bob")
    assert user.name == "bob"
    assert user.address is None


def test_enum_field_is_constrained_and_in_schema():
    """An enum property becomes a Literal: only listed values validate, and the
    compiled model's JSON schema carries the enum (so structured-output backends
    like the in-browser WebLLM grammar can constrain generation)."""
    datatypes = {
        "Classification": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": ["order", "feedback", "question", "other"],
                }
            },
        }
    }
    parser = SchemaParser(datatypes)
    models = parser.parse_all()
    Classification = models["Classification"]

    assert Classification(intent="order").intent == "order"
    with pytest.raises(ValueError):
        Classification(intent="not-a-listed-value")

    schema = Classification.model_json_schema()
    assert schema["properties"]["intent"]["enum"] == [
        "order",
        "feedback",
        "question",
        "other",
    ]


def test_optional_enum_field():
    """An enum can also be optional via required: false."""
    datatypes = {
        "Feedback": {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative"],
                    "required": False,
                }
            },
        }
    }
    parser = SchemaParser(datatypes)
    models = parser.parse_all()
    Feedback = models["Feedback"]

    assert Feedback().sentiment is None
    assert Feedback(sentiment="positive").sentiment == "positive"
    with pytest.raises(ValueError):
        Feedback(sentiment="neutral")


def test_empty_enum_raises_error():
    """An empty enum list is rejected with a helpful error."""
    datatypes = {
        "Bad": {"type": "object", "properties": {"x": {"type": "string", "enum": []}}}
    }
    parser = SchemaParser(datatypes)
    with pytest.raises(ValueError, match="'enum' must be a non-empty list"):
        parser.parse_all()


def test_unknown_property_attribute_is_rejected():
    datatypes = {
        "User": {
            "type": "object",
            "properties": {"name": {"type": "string", "pattern": "^a"}},
        }
    }
    parser = SchemaParser(datatypes)
    with pytest.raises(
        ValueError, match="Unknown attribute 'pattern' in property 'name'"
    ):
        parser.parse_all()


def test_unknown_type_attribute_is_rejected():
    datatypes = {"User": {"type": "object", "properties": {}, "title": "User"}}
    parser = SchemaParser(datatypes)
    with pytest.raises(ValueError, match="Unknown attribute 'title' in type 'User'"):
        parser.parse_all()


def test_inline_nested_object_becomes_a_model():
    datatypes = {
        "Order": {
            "type": "object",
            "properties": {
                "customer": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "vip": {"type": "boolean", "required": False},
                    },
                }
            },
        }
    }
    parser = SchemaParser(datatypes)
    models = parser.parse_all()

    order = models["Order"](customer={"name": "Ada"})
    assert order.customer.name == "Ada"
    assert order.customer.vip is None
    # Inline objects forbid unknown keys.
    with pytest.raises(Exception):
        models["Order"](customer={"name": "Ada", "unexpected": 1})


def test_string_length_constraints_are_applied():
    datatypes = {
        "Note": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "min_length": 2, "max_length": 4},
            },
        }
    }
    parser = SchemaParser(datatypes)
    models = parser.parse_all()

    assert models["Note"](text="abc").text == "abc"
    with pytest.raises(Exception):
        models["Note"](text="a")
    with pytest.raises(Exception):
        models["Note"](text="abcde")


def test_top_level_ref_creates_a_named_alias():
    datatypes = {
        "Person": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        },
        "Customer": {"$ref": "Person"},
    }
    parser = SchemaParser(datatypes)
    models = parser.parse_all()

    customer = models["Customer"](name="Ada")
    assert customer.name == "Ada"
    assert models["Customer"].__name__ == "Customer"
    assert issubclass(models["Customer"], models["Person"])
