"""Name-to-backend registries for LLM, embedding and RAG backends.

Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

A workflow names its backends with strings -- ``llm_model: openai/gpt-5.4-mini``
in YAML, a ``rag_query`` node naming a service -- so resolution has to happen
from a name. Until this module existed that name could only be one of five
values hard-coded in an ``if`` chain, which meant adding a backend required
editing Kaval.AI itself.

Three registries live here, all instances of the same :class:`Registry`:

- ``llm_providers`` -- resolves ``provider/model`` to a
  :class:`~kavalai.llm_clients.base_client.BaseLlmClient`
- ``embedding_providers`` -- resolves ``provider/model`` to a
  :class:`~kavalai.llm_clients.embeddings.BaseEmbeddingClient`
- ``rag_services`` -- resolves a plain name to a
  :class:`~kavalai.rag.base.BaseRagService`

Registration is deliberately explicit -- there is no scanning and no plugin
path. The set of available backends is discovered; the behaviour of any given
run is still passed, since an engine's ``client_factory`` and ``rag_services``
arguments both outrank these registries.

A registration ``Target`` is a class, a dotted path to one, or a factory
callable. ``DEFAULT_NAME`` is the name looked up when a workflow does not name
a backend explicitly; it matches ``BaseRagService``'s own default collection
and source identifiers. ``BUILTIN_LLM_PACKAGES`` names the PyPI distribution
behind each built-in LLM provider, so an import failure can say what to
install.
"""

import importlib
from typing import Any, Callable, Optional, Union

from loguru import logger

Target = Union[type, str, Callable[..., Any]]

DEFAULT_NAME = "default"


class RegistryError(ValueError):
    """Raised for duplicate registrations, unknown names and bad dotted paths.

    Subclasses :class:`ValueError` because that is what ``make_client`` and
    ``make_embedding_client`` raised for an unusable name before the registries
    existed, and callers catch it.
    """


def _import_dotted(path: str, registered_as: str) -> Any:
    """Import ``pkg.module.Object`` and return the object.

    The registered name is folded into the error, because a bare
    ``ModuleNotFoundError: mycorppackage`` raised from inside a node execution
    does not say which registration produced it.
    """
    if "." not in path:
        raise RegistryError(
            f"'{registered_as}' is registered as '{path}', which is not a "
            "dotted path -- expected 'package.module.ClassName'."
        )
    module_path, _, attribute = path.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError as error:
        raise RegistryError(
            f"'{registered_as}' is registered as '{path}' but '{module_path}' "
            f"could not be imported: {error}"
        ) from error
    try:
        return getattr(module, attribute)
    except AttributeError as error:
        raise RegistryError(
            f"'{registered_as}' is registered as '{path}' but '{module_path}' "
            f"has no attribute '{attribute}'."
        ) from error


class Registry:
    """A name-to-target registry with lazy dotted-path resolution.

    Names may be a bare provider (``"openai"``) or a full ``provider/model``
    pair (``"mycorp/mymodel"``). Lookup tries the exact string first and then
    the part before the first ``/``, so a model-specific registration wins over
    a provider-wide one without either having to know about the other.

    ``kind`` is the noun used in error messages ("LLM provider") and
    ``register_hint`` the public function that registers into this registry,
    so an error can name the call that fixes it. Names shipped with Kaval.AI
    are marked as built-ins; :meth:`verify` skips them, because their SDKs
    are optional extras that a given install need not have.
    """

    def __init__(self, kind: str, register_hint: str):
        self.kind = kind
        self.register_hint = register_hint
        self._targets: dict[str, Target] = {}
        self._resolved: dict[str, Any] = {}
        self._defaults: dict[str, dict[str, Any]] = {}
        self._builtins: set[str] = set()

    def register(
        self,
        name: str,
        target: Target,
        *,
        replace: bool = False,
        validate: bool = False,
        **defaults: Any,
    ) -> None:
        """Bind ``name`` to ``target``.

        Args:
            name: Provider name (``"mycorp"``) or ``provider/model`` pair.
            target: A class, a dotted path to one, or a factory callable.
            replace: Overwrite an existing registration. Without it a duplicate
                raises, matching ``FunctionKernel.register_python_tool``.
            validate: Import a dotted path immediately instead of on first use.
            **defaults: Bound now, forwarded to the target at construction.

        Raises:
            RegistryError: The name is empty, starts or ends with ``/``, or is
                already registered and ``replace`` is not set.
        """
        if not name:
            raise RegistryError(f"A {self.kind} name cannot be empty.")
        if name.startswith("/") or name.endswith("/"):
            raise RegistryError(
                f"Invalid {self.kind} name '{name}': '/' separates a provider "
                "from a model and cannot be the first or last character."
            )

        existing = self._targets.get(name)
        if existing is not None and not replace:
            raise RegistryError(
                f"{self.kind} '{name}' is already registered (as {existing!r}). "
                "Pass replace=True to override it."
            )
        if existing is not None:
            # Shadowing a working name changes what every later lookup means,
            # so it is worth a line in the log rather than silence.
            logger.warning(
                f"{self.kind} '{name}' re-registered: {existing!r} -> {target!r}."
            )

        self._targets[name] = target
        self._defaults[name] = defaults
        self._resolved.pop(name, None)

        if validate and isinstance(target, str):
            self._resolve_target(name)

    def unregister(self, name: str) -> None:
        """Remove a registration. Unknown names are ignored."""
        self._targets.pop(name, None)
        self._defaults.pop(name, None)
        self._resolved.pop(name, None)

    def names(self) -> list[str]:
        """Every registered name, sorted. Resolves nothing."""
        return sorted(self._targets)

    def _resolve_target(self, name: str) -> Any:
        """Return the callable registered under ``name``, importing if needed."""
        if name in self._resolved:
            return self._resolved[name]
        target = self._targets[name]
        if isinstance(target, str):
            target = _import_dotted(target, name)
        self._resolved[name] = target
        return target

    def lookup(self, name: str) -> Optional[str]:
        """Return the registered key serving ``name``, or ``None``.

        Tries the exact name first, then the provider part before the first
        ``/``. Nothing is imported.
        """
        if name in self._targets:
            return name
        provider, separator, _ = name.partition("/")
        if separator and provider in self._targets:
            return provider
        return None

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Construct the backend registered for ``name``.

        Classes are built through ``from_model``; anything else is called
        directly. Registration defaults are merged in beneath the caller's
        keyword arguments, so a caller can always override them.

        Raises:
            RegistryError: Nothing is registered for ``name``.
        """
        key = self.lookup(name)
        if key is None:
            raise RegistryError(self.unknown_message(name))

        target = self._resolve_target(key)
        merged = {**self._defaults.get(key, {}), **kwargs}
        try:
            if isinstance(target, type) and hasattr(target, "from_model"):
                return target.from_model(*args, **merged)
            return target(*args, **merged)
        except TypeError as error:
            # Registration defaults are plain kwargs, so a misspelled one
            # surfaces here as a TypeError from someone else's constructor.
            # Naming the registration turns that into a usable message.
            raise RegistryError(
                f"Could not construct {self.kind} '{name}' registered as "
                f"'{key}' ({target!r}) with {merged!r}: {error}"
            ) from error

    def unknown_message(self, name: str) -> str:
        """The error text for a name nothing is registered for."""
        provider, separator, _ = name.partition("/")
        tried = f"'{name}'"
        if separator:
            tried = f"'{name}' and '{provider}'"
        known = ", ".join(self.names()) or "(none)"
        return (
            f"Unsupported {self.kind} '{name}': tried {tried}. "
            f"Registered: {known}. Add your own with {self.register_hint}(), "
            "which must run before the workflow is loaded."
        )

    def verify(self) -> None:
        """Import every non-built-in dotted registration.

        Raises on the first that cannot be imported. Entry points call this
        after loading operator-supplied modules, so a mistyped path fails at
        start-up rather than at the first request that happens to reach that
        node.

        Built-ins are skipped deliberately: their SDKs are optional extras, and
        an install with only ``openai`` should not fail start-up because
        ``anthropic`` is absent.
        """
        for name, target in self._targets.items():
            if isinstance(target, str) and name not in self._builtins:
                self._resolve_target(name)

    def mark_builtin(self, name: str) -> None:
        """Record ``name`` as shipped with Kaval.AI, so ``verify`` skips it."""
        self._builtins.add(name)


llm_providers = Registry("LLM provider", "register_llm_provider")
embedding_providers = Registry("embedding provider", "register_embedding_provider")
rag_services = Registry("RAG service", "register_rag_service")


# Built-ins are registered as dotted paths rather than imported classes: the
# provider SDKs are optional extras, and ``import kavalai`` has to keep working
# under Pyodide where none of them can be installed.
_BUILTIN_LLM_PROVIDERS = {
    "openai": "kavalai.llm_clients.openai_client.OpenAIClient",
    "gemini": "kavalai.llm_clients.gemini_client.GeminiClient",
    "anthropic": "kavalai.llm_clients.anthropic_client.AnthropicClient",
    "ollama": "kavalai.llm_clients.ollama_client.OllamaClient",
    "browser": "kavalai.llm_clients.browser_client.BrowserLLMClient",
}

BUILTIN_LLM_PACKAGES = {
    "openai": "openai",
    "gemini": "google-genai",
    "anthropic": "anthropic",
    "ollama": "ollama",
}

_BUILTIN_EMBEDDING_PROVIDERS = {
    "openai": "kavalai.llm_clients.embeddings.OpenAIEmbeddingClient",
    "gemini": "kavalai.llm_clients.embeddings.GeminiEmbeddingClient",
    "ollama": "kavalai.llm_clients.embeddings.OllamaEmbeddingClient",
    "fastembed": "kavalai.llm_clients.embeddings.FastEmbedClient",
    "browser": "kavalai.llm_clients.embeddings.BrowserEmbeddingClient",
}

_BUILTIN_RAG_SERVICES = {
    "postgres": "kavalai.rag.postgres.PostgresRagService",
    "sqlite": "kavalai.rag.sqllite.SqliteRagService",
}

for _registry, _builtins in (
    (llm_providers, _BUILTIN_LLM_PROVIDERS),
    (embedding_providers, _BUILTIN_EMBEDDING_PROVIDERS),
    (rag_services, _BUILTIN_RAG_SERVICES),
):
    for _name, _path in _builtins.items():
        _registry.register(_name, _path)
        _registry.mark_builtin(_name)


def register_llm_provider(
    name: str,
    target: Target,
    *,
    replace: bool = False,
    validate: bool = False,
    **defaults: Any,
) -> None:
    """Register an LLM client under ``name``, usable as ``name/<model>``.

    ``name`` may also be a full ``provider/model`` pair, which pins one model
    to one class; lookup prefers it over a provider-wide registration.

    ::

        register_llm_provider("mycorp", MyClient)
        register_llm_provider("mycorp", "mycorp.llm.MyClient")
        register_llm_provider("mycorp/fast", MyClient, base_url="https://...")

    The target receives ``(model, parameters, stats_receiver)`` plus any
    registration defaults, where ``model`` is the part after the first ``/``.
    Forwarding ``stats_receiver`` to :class:`BaseLlmClient` is what makes a
    custom client's usage show up in the backoffice alongside the built-ins.
    """
    llm_providers.register(name, target, replace=replace, validate=validate, **defaults)


def register_embedding_provider(
    name: str,
    target: Target,
    *,
    replace: bool = False,
    validate: bool = False,
    **defaults: Any,
) -> None:
    """Register an embedding client under ``name``, usable as ``name/<model>``.

    The target receives ``(model,)`` plus any registration defaults.
    """
    embedding_providers.register(
        name, target, replace=replace, validate=validate, **defaults
    )


def register_rag_service(
    name: str,
    target: Target,
    *,
    replace: bool = False,
    validate: bool = False,
    **defaults: Any,
) -> None:
    """Register a RAG service under ``name``.

    Unlike models, a RAG service is named on its own -- there is no
    ``provider/model`` split. The name is what a ``rag_query`` node's
    ``service`` field and a workflow's ``rag_service`` default refer to;
    ``"default"`` is used when neither names one.

    The target receives only the registration defaults, so everything the
    backend needs (a URI, a filename, an embedding model) is bound here::

        register_rag_service(
            "default", SqliteRagService,
            filename="handbook.db", model="fastembed/BAAI/bge-small-en-v1.5",
        )
    """
    rag_services.register(name, target, replace=replace, validate=validate, **defaults)


def make_rag_service(name: str = DEFAULT_NAME, **kwargs: Any) -> Any:
    """Construct the RAG service registered under ``name``.

    Args:
        name: A registered service name. Defaults to ``"default"``, which is
            what a workflow uses when neither a node nor the graph names one.
        **kwargs: Extra constructor arguments, overriding any bound at
            registration.

    Raises:
        RegistryError: No service is registered under that name.
    """
    return rag_services.build(name, **kwargs)


def registered_llm_providers() -> list[str]:
    """Every registered LLM provider name, sorted."""
    return llm_providers.names()


def registered_embedding_providers() -> list[str]:
    """Every registered embedding provider name, sorted."""
    return embedding_providers.names()


def registered_rag_services() -> list[str]:
    """Every registered RAG service name, sorted."""
    return rag_services.names()


def verify_registrations() -> None:
    """Import every dotted registration in every registry.

    Raises :class:`RegistryError` on the first that cannot be imported. Called
    by entry points after loading operator-supplied provider modules, so a
    mistyped path fails at start-up instead of mid-run.

    Built-ins are skipped -- their SDKs are optional extras, so importing them
    all would make start-up depend on packages the install may not need.
    """
    llm_providers.verify()
    embedding_providers.verify()
    rag_services.verify()
