"""Parameter validation against JSON Schema for backtesting."""

from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


class ParamValidationError(Exception):
    """Error validating parameters against schema."""

    def __init__(self, message: str, errors: Optional[list[dict]] = None):
        super().__init__(message)
        self.message = message
        self.errors = errors or []


def validate_params(
    params: dict[str, Any],
    param_schema: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate parameters against a JSON Schema and apply defaults.

    Args:
        params: User-provided parameters
        param_schema: JSON Schema from compiled strategy spec

    Returns:
        Validated params with defaults applied

    Raises:
        ParamValidationError: If validation fails
    """
    if not param_schema:
        return params

    properties = param_schema.get("properties", {})
    required = param_schema.get("required", [])
    validated = {}
    errors = []

    # Check required params
    for req in required:
        if req not in params:
            errors.append({
                "param": req,
                "error": "required parameter missing",
            })

    # Validate each property
    for prop_name, prop_schema in properties.items():
        prop_type = prop_schema.get("type", "any")

        if prop_name in params:
            value = params[prop_name]

            # Type validation
            if prop_type == "integer":
                if not isinstance(value, int) or isinstance(value, bool):
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        errors.append({
                            "param": prop_name,
                            "error": f"expected integer, got {type(value).__name__}",
                            "value": value,
                        })
                        continue

            elif prop_type == "number":
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        errors.append({
                            "param": prop_name,
                            "error": f"expected number, got {type(value).__name__}",
                            "value": value,
                        })
                        continue

            elif prop_type == "string":
                if not isinstance(value, str):
                    value = str(value)

            elif prop_type == "boolean":
                if not isinstance(value, bool):
                    errors.append({
                        "param": prop_name,
                        "error": f"expected boolean, got {type(value).__name__}",
                        "value": value,
                    })
                    continue

            # Range validation (for numeric types)
            if prop_type in ("integer", "number"):
                if "minimum" in prop_schema and value < prop_schema["minimum"]:
                    errors.append({
                        "param": prop_name,
                        "error": f"value {value} below minimum {prop_schema['minimum']}",
                        "value": value,
                    })
                    continue

                if "maximum" in prop_schema and value > prop_schema["maximum"]:
                    errors.append({
                        "param": prop_name,
                        "error": f"value {value} above maximum {prop_schema['maximum']}",
                        "value": value,
                    })
                    continue

            # Enum validation
            if "enum" in prop_schema and value not in prop_schema["enum"]:
                errors.append({
                    "param": prop_name,
                    "error": f"value must be one of {prop_schema['enum']}",
                    "value": value,
                })
                continue

            validated[prop_name] = value

        elif "default" in prop_schema:
            # Apply default
            validated[prop_name] = prop_schema["default"]

    # Include any extra params not in schema (pass-through)
    for key, value in params.items():
        if key not in validated:
            validated[key] = value

    if errors:
        raise ParamValidationError(
            f"Parameter validation failed: {len(errors)} error(s)",
            errors=errors,
        )

    logger.debug(
        "Parameters validated",
        original_count=len(params),
        validated_count=len(validated),
    )

    return validated
