import functools
import re
from dataclasses import Field, fields
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    get_type_hints,
)

import fastjsonschema
import jsonschema
from dateutil.parser import parse
from mashumaro.config import (
    ADD_SERIALIZATION_CONTEXT,
    TO_DICT_ADD_OMIT_NONE_FLAG,
)
from mashumaro.config import (
    BaseConfig as MashBaseConfig,
)
from mashumaro.jsonschema import build_json_schema

# following includes DataClassDictMixin
from mashumaro.mixins.msgpack import DataClassMessagePackMixin
from mashumaro.types import SerializableType, SerializationStrategy


class ValidationError(jsonschema.ValidationError):
    pass


class DateTimeSerialization(SerializationStrategy):
    def serialize(self, value: datetime) -> str:
        out = value.isoformat()
        # Assume UTC if timezone is missing
        if value.tzinfo is None:
            out += "Z"
        return out

    def deserialize(self, value: Union[datetime, str]) -> datetime:
        return value if isinstance(value, datetime) else parse(value)


class dbtMashConfig(MashBaseConfig):
    code_generation_options = [
        TO_DICT_ADD_OMIT_NONE_FLAG,
        ADD_SERIALIZATION_CONTEXT,
    ]
    serialization_strategy = {
        datetime: DateTimeSerialization(),
    }
    json_schema = {
        "additionalProperties": False,
    }
    serialize_by_alias = True
    lazy_compilation = True


# fastjsonschema rejects unknown "format" values at compile time. The
# mashumaro-generated schemas don't currently use OpenAPI numeric format hints,
# but we mirror dbt-core/jsonschemas/jsonschemas.py defensively in case future
# schemas pick them up.
_NUMERIC_FORMAT_NOOPS: Dict[str, Callable[[Any], bool]] = {
    fmt: (lambda _: True) for fmt in ("int32", "int64", "uint64", "float", "double")
}

# Cache of compiled fastjsonschema validators keyed by id(json_schema). The
# json_schema dict for each dbtClassMixin subclass is memoized on the class via
# functools.lru_cache, so id() is stable for the process lifetime. A value of
# None marks a schema fastjsonschema could not compile (slow path always).
_FAST_VALIDATOR_CACHE: Dict[int, Optional[Callable[[Any], Any]]] = {}


def _get_fast_validator(schema: Dict[str, Any]) -> Optional[Callable[[Any], Any]]:
    key = id(schema)
    if key in _FAST_VALIDATOR_CACHE:
        return _FAST_VALIDATOR_CACHE[key]
    try:
        # use_default=False avoids fastjsonschema mutating the input dict by
        # injecting schema `default` values (it defaults to True). dbt schemas
        # declare `"default": null` for several optional fields, and the
        # downstream slow-path validator rejects None for the typed field.
        compiled = fastjsonschema.compile(
            schema,
            formats=_NUMERIC_FORMAT_NOOPS,
            use_default=False,
        )
    except Exception:
        _FAST_VALIDATOR_CACHE[key] = None
        return None
    _FAST_VALIDATOR_CACHE[key] = compiled
    return compiled


# This class pulls in DataClassDictMixin from Mashumaro. The 'to_dict'
# and 'from_dict' methods come from Mashumaro.
# Note: DataClassMessagePackMixin inherits from DataClassDictMixin
class dbtClassMixin(DataClassMessagePackMixin):
    """Convert and validate JSON schemas.

    The Mixin adds methods to generate a JSON schema and
    convert to and from JSON encodable dicts with validation
    against the schema
    """

    _mapped_fields: ClassVar[Optional[Dict[Any, List[Tuple[Field, str]]]]] = None

    # Config class used by Mashumaro
    class Config(dbtMashConfig):
        pass

    ADDITIONAL_PROPERTIES: ClassVar[bool] = False

    # This is called by the mashumaro from_dict in order to handle
    # nested classes. We no longer do any munging here, but leaving here
    # so that subclasses can leave super() in place for possible future needs.
    @classmethod
    def __pre_deserialize__(cls, data):
        return data

    # This is called by the mashumaro to_dict in order to handle
    # nested classes. We no longer do any munging here, but leaving here
    # so that subclasses can leave super() in place for possible future needs.
    def __post_serialize__(self, data, context: Optional[Dict]):
        return data

    @classmethod
    @functools.lru_cache
    def json_schema(cls):
        json_schema_obj = build_json_schema(cls)
        json_schema = json_schema_obj.to_dict()
        return json_schema

    @classmethod
    def validate(cls, data: Any) -> None:
        json_schema = cls.json_schema()
        # Fast path: try the compiled fastjsonschema validator first. On valid data
        # this is roughly 5x faster than jsonschema.Draft7Validator.iter_errors.
        # On invalid data it raises immediately; we then fall through to the slow
        # path, which is the only one that can produce a properly-typed
        # jsonschema.ValidationError for `ValidationError.create_from(...)`.
        fast = _get_fast_validator(json_schema)
        if fast is not None:
            try:
                fast(data)
            except fastjsonschema.JsonSchemaException:
                pass
            else:
                return
        validator = jsonschema.Draft7Validator(json_schema)
        error = next(iter(validator.iter_errors(data)), None)
        if error is not None:
            raise ValidationError.create_from(error) from error

    # This method was copied from hologram. Used in model_config.py and relation.py
    @classmethod
    def _get_fields(cls) -> List[Tuple[Field, str]]:
        if cls._mapped_fields is None:
            cls._mapped_fields = {}
        if cls.__name__ not in cls._mapped_fields:
            mapped_fields = []
            type_hints = get_type_hints(cls)

            for f in fields(cls):  # type: ignore
                # Skip internal fields
                if f.name.startswith("_"):
                    continue

                # Note fields() doesn't resolve forward refs
                f.type = type_hints[f.name]

                # hologram used the "field_mapping" here, but we use the
                # the field's metadata "alias". Since this method is mainly
                # just used in merging config dicts, it mostly applies to
                # pre-hook and post-hook.
                field_name = f.metadata.get("alias", f.name)
                mapped_fields.append((f, field_name))
            cls._mapped_fields[cls.__name__] = mapped_fields
        return cls._mapped_fields[cls.__name__]

    # copied from hologram. Used in tests
    @classmethod
    def _get_field_names(cls) -> List[str]:
        return [element[1] for element in cls._get_fields()]


class ValidatedStringMixin(str, SerializableType):
    ValidationRegex = ""

    @classmethod
    def _deserialize(cls, value: str) -> "ValidatedStringMixin":
        cls.validate(value)
        return ValidatedStringMixin(value)

    def _serialize(self) -> str:
        return str(self)

    @classmethod
    def validate(cls, value):
        res = re.match(cls.ValidationRegex, value)

        if res is None:
            raise ValidationError(f"Invalid value: {value}")  # TODO


# These classes must be in this order or it doesn't work
class StrEnum(str, SerializableType, Enum):
    def __str__(self) -> str:
        return self.value

    # https://docs.python.org/3.6/library/enum.html#using-automatic-values
    def _generate_next_value_(name, *_):
        return name

    def _serialize(self) -> str:
        return self.value

    @classmethod
    def _deserialize(cls, value: str):
        return cls(value)


class ExtensibleDbtClassMixin(dbtClassMixin):
    ADDITIONAL_PROPERTIES = True

    class Config(dbtMashConfig):
        json_schema = {
            "additionalProperties": True,
        }
