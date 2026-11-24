import inspect
from typing import Any, Literal, Optional

import pytest
from pydantic import BaseModel, ValidationError

from kavalai import pythontool
from kavalai.functionkernel import Validator
from kavalai.step_schema import (
    ArgRef,
    build_step_schema,
    has_closed_schema,
    step_output_model,
    typed_call_model,
    untyped_tools,
)

class Address(BaseModel):
    street: str
    city: str


class Weather(BaseModel):
    city: str
    unit: Literal["celsius", "fahrenheit"] = "celsius"
    days: Optional[int] = None


class Order(BaseModel):
    customer_id: str
    shipping: Address
    lines: list[Address]


class Search(BaseModel):
    query: str
    filters: dict


class Answer(BaseModel):
    text: str


def test_the_output_type_is_part_of_the_step():
    schema = build_step_schema(Answer, {}).model_json_schema()
    assert {"$ref": "#/$defs/Answer"} in schema["properties"]["output"]["anyOf"]
    step = build_step_schema(Answer, {}).model_validate(
        {"instructions": "i", "output": {"text": "done"}}
    )
    assert step.output == Answer(text="done")


def test_the_step_has_one_call_model_per_tool():
    schema = build_step_schema(
        str, {"python://weather": Weather, "rest://crm.find [GET]": Order}
    ).model_json_schema()
    options = schema["properties"]["tool_calls"]["items"]["anyOf"]
    assert [option["$ref"].rsplit("/", 1)[1] for option in options] == [
        "python_weather_call",
        "rest_crm_find_GET_call",
    ]
    assert schema["$defs"]["rest_crm_find_GET_call"]["properties"]["name"]["const"] == (
        "rest://crm.find [GET]"
    )
    assert schema["properties"]["instructions"]["description"]
    assert schema["required"] == ["instructions"]


def test_a_single_tool_needs_no_union():
    schema = build_step_schema(str, {"python://weather": Weather}).model_json_schema()
    assert schema["properties"]["tool_calls"]["items"] == {
        "$ref": "#/$defs/python_weather_call"
    }


def test_a_step_without_tools_has_no_tool_calls():
    model = build_step_schema(str, {})
    assert list(model.model_fields) == ["instructions", "output"]


def test_uris_that_reduce_to_one_name_stay_apart():
    schema = build_step_schema(
        str, {"python://a.b": Weather, "python://a_b": Weather, "python://a-b": Weather}
    ).model_json_schema()
    calls = sorted(name for name in schema["$defs"] if name.endswith("_call"))
    assert calls == ["python_a_b_2_call", "python_a_b_3_call", "python_a_b_call"]
    names = {
        schema["$defs"][call]["properties"]["name"]["const"] for call in calls
    }
    assert names == {"python://a.b", "python://a_b", "python://a-b"}


def test_typed_call_holds_arguments_to_the_tool():
    call = typed_call_model("python://weather", Weather)
    schema = call.model_json_schema()
    assert schema["additionalProperties"] is False
    args = schema["$defs"]["python_weather_args"]
    assert args["additionalProperties"] is False
    assert "required" not in args
    assert {"type": "null"} in args["properties"]["city"]["anyOf"]
    assert schema["$defs"]["python_weather_ref"]["properties"]["arg"]["enum"] == [
        "city",
        "unit",
        "days",
    ]

    call.model_validate({"name": "python://weather", "args": {"city": "Tartu"}})
    call.model_validate(
        {
            "name": "python://weather",
            "args": {"city": None, "unit": None, "days": None},
            "refs": [{"arg": "city", "source": "input", "key": "home"}],
            "call_id": "w",
        }
    )
    for wrong in (
        {"name": "python://weather", "args": {"city": "Tartu", "unit": "kelvin"}},
        {"name": "python://weather", "args": {"town": "Tartu"}},
        {"name": "python://other", "args": {}},
        {"name": "python://weather"},
        {
            "name": "python://weather",
            "args": {},
            "refs": [{"arg": "town", "source": "step", "key": "k"}],
        },
        {
            "name": "python://weather",
            "args": {},
            "refs": [{"arg": "city", "source": "memory", "key": "k"}],
        },
    ):
        with pytest.raises(ValidationError):
            call.model_validate(wrong)


def test_typed_call_keeps_parameter_descriptions():
    from pydantic import Field

    class Described(BaseModel):
        city: str = Field(description="The city, in English.")

    args = typed_call_model("python://w", Described).model_json_schema()["$defs"]["python_w_args"]
    assert args["properties"]["city"]["description"] == "The city, in English."


def test_typed_call_types_nested_models():
    call = typed_call_model("python://order", Order)
    call.model_validate(
        {
            "name": "python://order",
            "args": {
                "customer_id": "C-1",
                "shipping": {"street": "Pikk 1", "city": "Tallinn"},
                "lines": [{"street": "a", "city": "b"}],
            },
        }
    )
    with pytest.raises(ValidationError):
        call.model_validate(
            {"name": "python://order", "args": {"shipping": {"street": "Pikk 1"}}}
        )


def test_typed_call_of_a_tool_without_parameters_has_no_refs():
    @pythontool
    def ping() -> str:
        return "pong"

    assert ping() == "pong"
    input_model = Validator.create_model_from_signature("ping", inspect.signature(ping))
    call = typed_call_model("python://ping", input_model)
    assert list(call.model_fields) == ["name", "args", "call_id"]
    call.model_validate({"name": "python://ping", "args": {}})


def test_a_tool_with_a_free_form_argument_takes_a_json_string():
    call = typed_call_model("mcp://kb.search", Search)
    assert list(call.model_fields) == ["name", "args_json", "refs", "call_id"]
    schema = call.model_json_schema()
    assert schema["properties"]["args_json"]["type"] == "string"
    assert schema["$defs"]["mcp_kb_search_ref"]["properties"]["arg"]["enum"] == [
        "query",
        "filters",
    ]
    call.model_validate({"name": "mcp://kb.search", "args_json": '{"query": "x"}'})
    with pytest.raises(ValidationError):
        call.model_validate({"name": "mcp://kb.search", "args": {"query": "x"}})


def test_a_custom_stem_names_the_models():
    call = typed_call_model("python://weather", Weather, stem="w")
    assert call.__name__ == "w_call"
    assert "w_args" in call.model_json_schema()["$defs"]


def model_of(annotation: Any) -> type[BaseModel]:
    from pydantic import create_model

    return create_model("M", value=(annotation, ...))


@pytest.mark.parametrize(
    "annotation",
    [
        str,
        int,
        Optional[bool],
        Literal["a", "b"],
        list[str],
        list[Address],
        tuple[int, str],
        Address,
        Optional[Address],
    ],
)
def test_closed_schemas(annotation):
    assert has_closed_schema(model_of(annotation))


@pytest.mark.parametrize(
    "annotation",
    [dict, Any, list, dict[str, int], Optional[dict], list[dict], list[Any], Search],
)
def test_free_form_schemas(annotation):
    assert not has_closed_schema(model_of(annotation))


def test_a_model_without_fields_is_closed():
    from pydantic import create_model

    assert has_closed_schema(create_model("Empty"))


def test_untyped_tools_are_listed_in_order():
    tools = {"python://weather": Weather, "mcp://kb.search": Search, "python://o": Order}
    assert untyped_tools(tools) == ["mcp://kb.search"]
    assert untyped_tools({}) == []


def test_step_output_model_takes_any_call_type():
    model = step_output_model(str, ArgRef)
    step = model.model_validate(
        {"instructions": "i", "tool_calls": [{"arg": "a", "source": "step", "key": "k"}]}
    )
    assert step.tool_calls[0] == ArgRef(arg="a", source="step", key="k")
    assert step.output is None


def test_arg_ref_names_its_source():
    with pytest.raises(ValidationError):
        ArgRef(arg="a", source="memory", key="k")


def test_the_module_imports_pydantic_only():
    """It has to load under Pyodide, before any provider SDK or database."""
    import kavalai.step_schema as module

    imported = [
        line.split()[1]
        for line in inspect.getsource(module).splitlines()
        if line.startswith(("import ", "from "))
    ]
    assert imported == ["re", "typing", "pydantic"]
