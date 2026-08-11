import json

import pytest

from kavalai.tools.openapi_spec_parser import (
    OpenApiSpecParser,
)  # Assuming your class is in openapi_parser.py

EXAMPLE_SPEC = """
{
  "openapi": "3.1.0",
  "info": {
    "title": "Johnny Silverhand agent",
    "description": "Basic configuration example of a chatbot with personality.",
    "version": "0.1.0"
  },
  "paths": {
    "/run_agent": {
      "post": {
        "summary": "Run Agent",
        "operationId": "run_agent",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/InputType"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/OutputType"
                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        },
        "security": [
          {
            "HTTPBasic": []
          }
        ]
      }
    }
  },
  "components": {
    "schemas": {
      "HTTPValidationError": {
        "properties": {
          "detail": {
            "items": {
              "$ref": "#/components/schemas/ValidationError"
            },
            "type": "array",
            "title": "Detail"
          }
        },
        "type": "object",
        "title": "HTTPValidationError"
      },
      "InputType": {
        "properties": {
          "session_id": {
            "anyOf": [
              {
                "type": "string",
                "format": "uuid"
              },
              {
                "type": "null"
              }
            ],
            "title": "Session Id"
          },
          "external_id": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "External Id"
          },
          "data": {
            "$ref": "#/components/schemas/input"
          }
        },
        "type": "object",
        "required": [
          "data"
        ],
        "title": "InputType"
      },
      "OutputType": {
        "properties": {
          "session_id": {
            "anyOf": [
              {
                "type": "string",
                "format": "uuid"
              },
              {
                "type": "null"
              }
            ],
            "title": "Session Id"
          },
          "data": {
            "$ref": "#/components/schemas/output"
          }
        },
        "type": "object",
        "required": [
          "session_id",
          "data"
        ],
        "title": "OutputType"
      },
      "ValidationError": {
        "properties": {
          "loc": {
            "items": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "integer"
                }
              ]
            },
            "type": "array",
            "title": "Location"
          },
          "msg": {
            "type": "string",
            "title": "Message"
          },
          "type": {
            "type": "string",
            "title": "Error Type"
          }
        },
        "type": "object",
        "required": [
          "loc",
          "msg",
          "type"
        ],
        "title": "ValidationError"
      },
      "input": {
        "properties": {
          "user_message": {
            "type": "string",
            "title": "User Message"
          }
        },
        "type": "object",
        "required": [
          "user_message"
        ],
        "title": "input"
      },
      "output": {
        "properties": {
          "agent_response": {
            "type": "string",
            "maxLength": 100,
            "title": "Agent Response"
          }
        },
        "type": "object",
        "required": [
          "agent_response"
        ],
        "title": "output"
      }
    },
    "securitySchemes": {
      "HTTPBasic": {
        "type": "http",
        "scheme": "basic"
      }
    }
  }
}
"""


@pytest.fixture()
def example_spec():
    return json.loads(EXAMPLE_SPEC)


def test_resolution(example_spec):
    """Test that a basic $ref is replaced by its content."""
    parser = OpenApiSpecParser(example_spec)
    assert parser.get_path_request_schema("/run_agent", "POST")
    assert parser.get_path_response_schema("/run_agent", "POST")


def test_external_references_are_reported_not_resolved():
    parser = OpenApiSpecParser({"openapi": "3.1.0"})
    resolved = parser._get_referenced_data("https://example.com/schema.json#/Foo")
    assert "not supported" in resolved["error"]


def test_pointer_can_index_into_a_list():
    spec = {"servers": [{"url": "http://a"}, {"url": "http://b"}]}
    parser = OpenApiSpecParser(spec)
    assert parser._get_referenced_data("#/servers/1") == {"url": "http://b"}


def test_unresolvable_pointer_raises_key_error():
    parser = OpenApiSpecParser({"components": {}})
    with pytest.raises(KeyError, match="Could not resolve pointer part"):
        parser._get_referenced_data("#/components/schemas/Missing")


def test_circular_references_stop_recursing():
    spec = {
        "components": {
            "schemas": {
                "Node": {
                    "type": "object",
                    "properties": {"parent": {"$ref": "#/components/schemas/Node"}},
                }
            }
        }
    }
    parser = OpenApiSpecParser(spec)
    node = parser.resolved_spec["components"]["schemas"]["Node"]
    parent = node["properties"]["parent"]
    # One level is expanded; the self-reference below it is left as a bare
    # $ref instead of recursing forever.
    assert parent["type"] == "object"
    assert parent["properties"]["parent"] == {"$ref": "#/components/schemas/Node"}
