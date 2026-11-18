"""Live LLM provider adapter supporting Local 2B models (LM Studio / Ollama), Gemini, and OpenAI APIs."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import httpx

from core.actions.domain import Domain
from core.contracts import FactSchema, TaskProposalSchema
from core.domains import registry as default_registry
from core.logger import get_logger
from interpretation.llm.base import BaseLLMProvider
from interpretation.llm.mock import MockLLMProvider
from interpretation.router import DomainRouter, RoutingOutcome, build_domain_system_prompt

logger = get_logger("live_llm")

# Preserved default system prompt for blocks world for backwards compatibility
SYSTEM_FORMALIZER_PROMPT = build_domain_system_prompt("blocks_world")


class LiveLLMProvider(BaseLLMProvider):
    """Integrates with local 2B models (LM Studio / Ollama) or cloud LLM endpoints (Gemini / OpenAI)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "auto",
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        fallback_provider: Optional[BaseLLMProvider] = None,
    ) -> None:
        # Dynamically reload .env so changes made by user take effect immediately without server restart
        try:
            from dotenv import load_dotenv
            from pathlib import Path
            env_file = Path(__file__).resolve().parent.parent.parent / ".env"
            if env_file.exists():
                load_dotenv(dotenv_path=env_file, override=True)
        except Exception:
            pass

        # Check LLM_PROVIDER in env if provider is "auto" or "default"
        env_llm_provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
        if (not provider or provider in ("auto", "default")) and env_llm_provider:
            self.provider = env_llm_provider
        else:
            self.provider = (provider or "auto").lower()
            if self.provider == "default":
                self.provider = "auto"

        self.model = model
        self.lm_studio_url = os.environ.get("LM_STUDIO_URL", "http://127.0.0.1:1234").rstrip("/")
        self.endpoint = endpoint or os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/v1")
        if api_key:
            self.api_key = api_key
        elif self.provider in ("gemini",):
            self.api_key = os.environ.get("GEMINI_API_KEY")
        elif self.provider in ("nvidia",):
            self.api_key = os.environ.get("NVIDIA_API_KEY")
        elif self.provider in ("openrouter",):
            self.api_key = os.environ.get("OPENROUTER_API_KEY")
        elif self.provider in ("openai",):
            self.api_key = os.environ.get("OPENAI_API_KEY")
        else:
            self.api_key = (
                os.environ.get("NVIDIA_API_KEY")
                or os.environ.get("OPENROUTER_API_KEY")
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )
        self.fallback = fallback_provider or MockLLMProvider()
        self.registry = getattr(fallback_provider, "registry", default_registry)
        self.router = DomainRouter(registry=self.registry)
        # Request cache for the ₹0 guarantee (Feature 17 in explainer doc)
        self._cache: Dict[str, TaskProposalSchema] = {}

    def interpret(self, prompt: str, domain: Optional[Domain] = None) -> TaskProposalSchema:
        """Interprets a natural language instruction into a formal TaskProposalSchema."""
        if domain is None:
            routing = self.router.route(prompt)
            if routing.outcome == RoutingOutcome.SUPPORTED and routing.selected_domain_id:
                domain = self.registry.get(routing.selected_domain_id)
            if domain is None:
                from core.actions.domain import create_blocks_world_domain
                domain = create_blocks_world_domain()

        domain_id = getattr(domain, "domain_id", getattr(domain, "name", "blocks_world")).lower()
        cache_key = f"{self.provider}:{self.model}:{domain_id}:{prompt.strip().lower()}"
        if cache_key in self._cache:
            logger.info(f"Returning cached formalization for '{prompt[:40]}...'")
            return self._cache[cache_key]

        # 1. Deterministic zero-API mock requested
        if self.provider == "mock":
            res = self.fallback.interpret(prompt, domain)
            self._cache[cache_key] = res
            return res

        # 2. NVIDIA NIM API (if explicitly selected or NVIDIA_API_KEY is configured)
        system_prompt = build_domain_system_prompt(domain, self.registry)
        nvidia_key = os.environ.get("NVIDIA_API_KEY")
        if (self.provider in ("nvidia", "auto")) and nvidia_key:
            try:
                proposal = self._call_nvidia(
                    prompt=prompt,
                    api_key=nvidia_key,
                    model=self.model or os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b"),
                    domain=domain,
                    system_prompt=system_prompt,
                )
                if proposal:
                    logger.info(f"Successfully formalized prompt via NVIDIA NIM: {prompt[:40]}...")
                    self._cache[cache_key] = proposal
                    return proposal
            except Exception as e:
                logger.warning(f"NVIDIA API invocation failed ({e}); falling back.")

        # 3. OpenRouter API (if explicitly selected or OPENROUTER_API_KEY is configured)
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if (self.provider in ("openrouter", "auto")) and openrouter_key:
            try:
                proposal = self._call_openrouter(
                    prompt=prompt,
                    api_key=openrouter_key,
                    model=self.model or os.environ.get("OPENROUTER_MODEL", "qwen/qwen-2.5-7b-instruct"),
                    domain=domain,
                    system_prompt=system_prompt,
                )
                if proposal:
                    logger.info(f"Successfully formalized prompt via OpenRouter: {prompt[:40]}...")
                    self._cache[cache_key] = proposal
                    return proposal
            except Exception as e:
                logger.warning(f"OpenRouter API invocation failed ({e}); falling back.")

        # 4. LM Studio Local Model (Native v1 /api/v1/chat or /v1/chat/completions)
        if self.provider in ("lmstudio", "lm_studio", "local_lm", "local", "auto"):
            try:
                proposal = self._call_lm_studio(prompt=prompt, domain=domain, system_prompt=system_prompt)
                if proposal:
                    logger.info(f"Successfully formalized prompt via LM Studio: {prompt[:40]}...")
                    self._cache[cache_key] = proposal
                    return proposal
            except Exception as e:
                logger.debug(f"LM Studio call failed ({e}); evaluating other providers or fallback.")

        # 5. Local Ollama 2B Model
        if self.provider in ("ollama", "local", "auto"):
            try:
                proposal = self._call_openai_compatible(
                    prompt=prompt,
                    endpoint=f"{self.endpoint.rstrip('/')}/chat/completions",
                    model=self.model or "gemma2:2b",
                    api_key="ollama",
                    domain=domain,
                    system_prompt=system_prompt,
                )
                if proposal:
                    logger.info(f"Successfully formalized prompt via Ollama: {prompt[:40]}...")
                    self._cache[cache_key] = proposal
                    return proposal
            except Exception as e:
                logger.debug(f"Ollama call failed ({e}); checking cloud or fallback.")

        # 6. Google Gemini API (if GEMINI_API_KEY is provided)
        gemini_key = self.api_key if self.provider == "gemini" else os.environ.get("GEMINI_API_KEY")
        if (self.provider in ("gemini", "auto")) and gemini_key:
            try:
                proposal = self._call_gemini_api(prompt=prompt, api_key=gemini_key, domain=domain, system_prompt=system_prompt)
                if proposal:
                    logger.info(f"Successfully formalized prompt via Gemini API: {prompt[:40]}...")
                    self._cache[cache_key] = proposal
                    return proposal
            except Exception as e:
                logger.warning(f"Gemini API invocation failed ({e}); falling back.")

        # 7. OpenAI API (if OPENAI_API_KEY is provided)
        openai_key = os.environ.get("OPENAI_API_KEY")
        if (self.provider in ("openai", "auto")) and openai_key:
            try:
                proposal = self._call_openai_compatible(
                    prompt=prompt,
                    endpoint="https://api.openai.com/v1/chat/completions",
                    model=self.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                    api_key=openai_key,
                    domain=domain,
                    system_prompt=system_prompt,
                )
                if proposal:
                    logger.info(f"Successfully formalized prompt via OpenAI: {prompt[:40]}...")
                    self._cache[cache_key] = proposal
                    return proposal
            except Exception as e:
                logger.warning(f"OpenAI API invocation failed ({e}); falling back.")

        # 8. Fallback to deterministic zero-API rule-based mock
        logger.info(f"Using deterministic zero-API fallback for prompt: '{prompt}'")
        res = self.fallback.interpret(prompt, domain)
        self._cache[cache_key] = res
        return res

    def interpret_with_retry(
        self,
        prompt: str,
        domain: Optional[Domain] = None,
        max_retries: int = 3,
        validation_errors: Optional[List[str]] = None,
    ) -> Tuple[TaskProposalSchema, str]:
        """Interprets with retry-on-error feedback. Returns (proposal, provider_name).

        If validation_errors is provided, augments the prompt with error feedback
        so the LLM can self-correct its output.
        """
        if domain is None:
            routing = self.router.route(prompt)
            if routing.outcome == RoutingOutcome.SUPPORTED and routing.selected_domain_id:
                domain = self.registry.get(routing.selected_domain_id)
            if domain is None:
                from core.actions.domain import create_blocks_world_domain
                domain = create_blocks_world_domain()

        system_prompt = build_domain_system_prompt(domain, self.registry)
        effective_prompt = prompt
        if validation_errors:
            error_feedback = "\n".join(f"- {e}" for e in validation_errors)
            effective_prompt = (
                f"{prompt}\n\n"
                f"IMPORTANT CORRECTION: Your previous formalization had these errors:\n"
                f"{error_feedback}\n"
                f"Please fix these issues in your JSON output."
            )

        # Try each provider and track which one succeeded
        if self.provider == "mock":
            return self.fallback.interpret(effective_prompt, domain), "mock"

        # NVIDIA NIM
        nvidia_key = os.environ.get("NVIDIA_API_KEY")
        if (self.provider in ("nvidia", "auto")) and nvidia_key:
            try:
                model_used = self.model or os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
                proposal = self._call_nvidia(effective_prompt, nvidia_key, model_used, domain=domain, system_prompt=system_prompt)
                if proposal:
                    return proposal, f"nvidia:{model_used}"
            except Exception as e:
                logger.debug(f"NVIDIA retry attempt failed: {e}")

        # OpenRouter
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if (self.provider in ("openrouter", "auto")) and openrouter_key:
            try:
                model_used = self.model or os.environ.get("OPENROUTER_MODEL", "qwen/qwen-2.5-7b-instruct")
                proposal = self._call_openrouter(effective_prompt, openrouter_key, model_used, domain=domain, system_prompt=system_prompt)
                if proposal:
                    return proposal, f"openrouter:{model_used}"
            except Exception as e:
                logger.debug(f"OpenRouter retry attempt failed: {e}")

        # LM Studio
        if self.provider in ("lmstudio", "lm_studio", "local_lm", "local", "auto"):
            try:
                proposal = self._call_lm_studio(effective_prompt, domain=domain, system_prompt=system_prompt)
                if proposal:
                    return proposal, f"lmstudio:{self.model or 'auto'}"
            except Exception as e:
                logger.debug(f"LM Studio retry attempt failed: {e}")

        # Ollama
        if self.provider in ("ollama", "local", "auto"):
            try:
                proposal = self._call_openai_compatible(
                    prompt=effective_prompt,
                    endpoint=f"{self.endpoint.rstrip('/')}/chat/completions",
                    model=self.model or "gemma2:2b",
                    api_key="ollama",
                    domain=domain,
                    system_prompt=system_prompt,
                )
                if proposal:
                    return proposal, f"ollama:{self.model or 'gemma2:2b'}"
            except Exception as e:
                logger.debug(f"Ollama retry attempt failed: {e}")

        # Gemini
        gemini_key = self.api_key if self.provider == "gemini" else os.environ.get("GEMINI_API_KEY")
        if (self.provider in ("gemini", "auto")) and gemini_key:
            try:
                proposal = self._call_gemini_api(prompt=effective_prompt, api_key=gemini_key, domain=domain, system_prompt=system_prompt)
                if proposal:
                    return proposal, f"gemini:{self.model or os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"
            except Exception as e:
                logger.debug(f"Gemini retry attempt failed: {e}")

        # OpenAI
        openai_key = os.environ.get("OPENAI_API_KEY")
        if (self.provider in ("openai", "auto")) and openai_key:
            try:
                proposal = self._call_openai_compatible(
                    prompt=effective_prompt,
                    endpoint="https://api.openai.com/v1/chat/completions",
                    model=self.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                    api_key=openai_key,
                    domain=domain,
                    system_prompt=system_prompt,
                )
                if proposal:
                    return proposal, f"openai:{self.model or 'gpt-4o-mini'}"
            except Exception as e:
                logger.debug(f"OpenAI retry attempt failed: {e}")

        # Fallback
        return self.fallback.interpret(effective_prompt, domain), "mock_fallback"

    def _call_lm_studio(
        self,
        prompt: str,
        domain: Optional[Domain] = None,
        system_prompt: Optional[str] = None,
    ) -> Optional[TaskProposalSchema]:
        """Communicates with LM Studio via Native v1 REST API (/api/v1/chat) or OpenAI-compatible endpoint."""
        base_url = self.lm_studio_url
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        sys_prompt = system_prompt or (build_domain_system_prompt(domain, self.registry) if domain else SYSTEM_FORMALIZER_PROMPT)

        # Fast connect-check to verify LM Studio is actively listening on port
        timeout = httpx.Timeout(120.0, connect=5.0)
        with httpx.Client(trust_env=False, timeout=timeout) as client:
            detected_model = self.model
            try:
                # First query LM Studio Native v1 API to check loaded instances
                v1_resp = client.get(f"{base_url}/api/v1/models", headers=headers)
                if v1_resp.status_code == 200:
                    models_list = v1_resp.json().get("models", [])
                    for m in models_list:
                        if m.get("loaded_instances") and not detected_model:
                            detected_model = m.get("key")
                            logger.info(f"Detected loaded LM Studio model: {detected_model}")
                            break
                if not detected_model:
                    # Fallback to OpenAI-compatible /v1/models
                    ping_resp = client.get(f"{base_url}/v1/models", headers=headers)
                    if ping_resp.status_code == 200:
                        m_data = ping_resp.json()
                        data_items = m_data.get("data", [])
                        if data_items and "id" in data_items[0]:
                            detected_model = data_items[0]["id"]
            except (httpx.ConnectTimeout, httpx.ConnectError, Exception) as e:
                logger.debug(f"LM Studio local server not running at {base_url} ({e}); skipping.")
                return None

            # 1. Primary: OpenAI-compatible endpoint with assistant thinking-bypass prefill
            openai_url = f"{base_url}/v1/chat/completions"
            openai_payload = {
                "model": detected_model or "local-model",
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "</think>```json\n"},
                ],
                "temperature": 0.0,
                "max_tokens": 512,
            }
            try:
                resp = client.post(openai_url, json=openai_payload, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    msg = data["choices"][0]["message"]
                    content = msg.get("content") or ""
                    # If model returned reasoning_content instead of content
                    if not content and "reasoning_content" in msg:
                        content = msg["reasoning_content"]
                    if content:
                        return self._parse_json_to_schema(content, prompt, domain=domain)
            except Exception as e:
                logger.debug(f"LM Studio /v1/chat/completions failed ({e}); attempting native endpoint.")

            # 2. Fallback: LM Studio Native v1 REST API (/api/v1/chat)
            native_url = f"{base_url}/api/v1/chat"
            native_payload = {
                "model": detected_model,
                "input": prompt,
                "system_prompt": sys_prompt,
                "temperature": 0.0,
            }
            try:
                resp = client.post(native_url, json=native_payload, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("content") or (data.get("message") or {}).get("content")
                    if content:
                        return self._parse_json_to_schema(content, prompt, domain=domain)
            except Exception as e:
                logger.debug(f"LM Studio native v1 API failed: {e}")

        return None

    def _call_gemini_api(
        self,
        prompt: str,
        api_key: str,
        domain: Optional[Domain] = None,
        system_prompt: Optional[str] = None,
    ) -> Optional[TaskProposalSchema]:
        """Calls official Google Gemini API to formalize natural language into JSON schema."""
        sys_prompt = system_prompt or (build_domain_system_prompt(domain, self.registry) if domain else SYSTEM_FORMALIZER_PROMPT)
        preferred_model = self.model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        candidate_models = [preferred_model]
        for fallback in ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.5-flash-lite"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        for model_name in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": sys_prompt},
                            {"text": f"Task instruction: {prompt}"},
                        ]
                    }
                ],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0.0,
                },
            }
            # Try both trust_env=True and trust_env=False to bypass restrictive Windows local proxies
            for trust in [True, False]:
                try:
                    with httpx.Client(timeout=20.0, trust_env=trust) as client:
                        resp = client.post(url, json=payload)
                        if resp.status_code == 200:
                            data = resp.json()
                            text = data["candidates"][0]["content"]["parts"][0]["text"]
                            return self._parse_json_to_schema(text, prompt, domain=domain)
                        elif resp.status_code == 404:
                            logger.info(f"Gemini model '{model_name}' returned 404; trying fallback.")
                            break
                        else:
                            logger.warning(f"Gemini API returned status {resp.status_code}: {resp.text[:200]}")
                            break
                except Exception as e:
                    logger.debug(f"Gemini API attempt (model={model_name}, trust_env={trust}) failed: {e}")
        return None

    def _call_nvidia(
        self,
        prompt: str,
        api_key: str,
        model: str,
        domain: Optional[Domain] = None,
        system_prompt: Optional[str] = None,
    ) -> Optional[TaskProposalSchema]:
        """Calls NVIDIA NIM chat completions API (https://integrate.api.nvidia.com/v1) with streaming."""
        sys_prompt = system_prompt or (build_domain_system_prompt(domain, self.registry) if domain else SYSTEM_FORMALIZER_PROMPT)
        base_url = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 4096,
            "stream": True,
        }

        for trust in [False, True]:
            try:
                accumulated_content = []
                with httpx.Client(trust_env=trust, timeout=90.0) as client:
                    with client.stream("POST", f"{base_url}/chat/completions", json=payload, headers=headers) as resp:
                        if resp.status_code != 200:
                            logger.warning(f"NVIDIA API returned HTTP {resp.status_code}: {resp.text[:300] if hasattr(resp, 'text') else ''}")
                            break
                        for line in resp.iter_lines():
                            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                                try:
                                    chunk = json.loads(line[6:])
                                    delta = chunk["choices"][0].get("delta", {})
                                    content_piece = delta.get("content", "")
                                    if content_piece:
                                        accumulated_content.append(content_piece)
                                except Exception:
                                    pass
                full_text = "".join(accumulated_content)
                if full_text:
                    return self._parse_json_to_schema(full_text, prompt, domain=domain)
            except Exception as e:
                logger.debug(f"NVIDIA NIM call attempt failed (trust_env={trust}): {e}")
        return None

    def _call_openrouter(
        self,
        prompt: str,
        api_key: str,
        model: str,
        domain: Optional[Domain] = None,
        system_prompt: Optional[str] = None,
    ) -> Optional[TaskProposalSchema]:
        """Calls OpenRouter chat completions API (https://openrouter.ai/api/v1/chat/completions)."""
        sys_prompt = system_prompt or (build_domain_system_prompt(domain, self.registry) if domain else SYSTEM_FORMALIZER_PROMPT)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}",
            "HTTP-Referer": "https://github.com/re-plan-v",
            "X-Title": "RE-PLAN-V Neurosymbolic",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 1024,
        }
        # Try both trust_env=False and trust_env=True for Windows proxy bypass
        for trust in [False, True]:
            try:
                with httpx.Client(trust_env=trust, timeout=60.0) as client:
                    resp = client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        content = data["choices"][0]["message"]["content"]
                        return self._parse_json_to_schema(content, prompt, domain=domain)
                    else:
                        logger.warning(
                            f"OpenRouter API returned {resp.status_code}: {resp.text[:300]}"
                        )
                        break
            except Exception as e:
                logger.debug(f"OpenRouter attempt failed (trust_env={trust}): {e}")
        return None

    def _call_openai_compatible(
        self,
        prompt: str,
        endpoint: str,
        model: str,
        api_key: str,
        domain: Optional[Domain] = None,
        system_prompt: Optional[str] = None,
    ) -> Optional[TaskProposalSchema]:
        """Calls standard OpenAI-compatible completions API."""
        sys_prompt = system_prompt or (build_domain_system_prompt(domain, self.registry) if domain else SYSTEM_FORMALIZER_PROMPT)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "</think>```json\n"},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        }
        is_local = "127.0.0.1" in endpoint or "localhost" in endpoint
        with httpx.Client(trust_env=not is_local, timeout=45.0) as client:
            resp = client.post(endpoint, json=payload, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return self._parse_json_to_schema(content, prompt, domain=domain)
        return None

    def _parse_json_to_schema(
        self,
        raw_json_str: str,
        prompt: str,
        domain: Optional[Domain] = None,
    ) -> TaskProposalSchema:
        """Parses model output JSON into a strongly typed, domain-aware TaskProposalSchema."""
        clean_json = raw_json_str.strip()
        match = re.search(r"(\{.*\})", clean_json, re.DOTALL)
        if match:
            clean_json = match.group(1)
        parsed = json.loads(clean_json)

        domain_id = getattr(domain, "domain_id", getattr(domain, "name", "blocks_world")).lower() if domain else "blocks_world"
        is_blocks = domain_id in ("blocks_world", "blocks", "tabletop")

        valid_types: Set[str] = set()
        if domain and hasattr(domain, "type_hierarchy"):
            valid_types = {t.lower() for t in domain.type_hierarchy.get_all_types() if t != "object"}
        if not valid_types:
            valid_types = {"block"}

        # Normalize entities
        entities_raw = parsed.get("entities", {})
        entities: Dict[str, str] = {}
        if isinstance(entities_raw, dict):
            for k, v in entities_raw.items():
                norm_k = str(k).strip().replace(" ", "_")
                norm_v = str(v).strip().replace(" ", "_").lower()
                if norm_v in valid_types:
                    entities[norm_k] = norm_v
                elif is_blocks:
                    entities[norm_k] = "block"
                else:
                    entities[norm_k] = norm_v
        elif isinstance(entities_raw, list):
            for item in entities_raw:
                if isinstance(item, dict):
                    name = str(item.get("value") or item.get("name") or str(item)).strip().replace(" ", "_")
                    etype = str(item.get("type", "block")).strip().lower()
                    entities[name] = etype if etype in valid_types else ("block" if is_blocks else sorted(valid_types)[0] if valid_types else "object")
                elif isinstance(item, str):
                    entities[item.strip().replace(" ", "_")] = "block" if is_blocks else (sorted(valid_types)[0] if valid_types else "object")

        def _parse_facts(raw_list: Any) -> List[FactSchema]:
            facts: List[FactSchema] = []
            if not isinstance(raw_list, list):
                return facts
            for f in raw_list:
                if isinstance(f, dict) and "predicate" in f and "arguments" in f:
                    pred = str(f["predicate"]).strip().rstrip("()").strip()
                    args = tuple(str(a).strip().replace(" ", "_").rstrip("()") for a in f["arguments"])
                    facts.append(FactSchema(predicate=pred, arguments=args))
                elif isinstance(f, str):
                    clean_str = f.strip()
                    m = re.match(r"^(\w+)\(([^)]*)\)$", clean_str)
                    if m:
                        pred = m.group(1).strip().rstrip("()").strip()
                        args = tuple(a.strip().replace(" ", "_").rstrip("()") for a in m.group(2).split(",") if a.strip())
                        facts.append(FactSchema(predicate=pred, arguments=args))
                    elif is_blocks:
                        if clean_str.rstrip("()") == "handempty":
                            facts.append(FactSchema(predicate="handempty", arguments=()))
                        elif "on_table" in clean_str or "table" in clean_str:
                            for token in clean_str.lower().split():
                                if "box" in token or "block" in token:
                                    facts.append(FactSchema(predicate="on_table", arguments=(token.strip(".,;:()").replace(" ", "_"),)))
                                    break
                        elif "clear" in clean_str:
                            for token in clean_str.lower().split():
                                if "box" in token or "block" in token:
                                    facts.append(FactSchema(predicate="clear", arguments=(token.strip(".,;:()").replace(" ", "_"),)))
                                    break
            return facts

        raw_initial_facts = _parse_facts(parsed.get("initial_facts", []))
        raw_goal_facts = _parse_facts(parsed.get("goal_facts", []))
        raw_negative_constraints = _parse_facts(parsed.get("negative_constraints", []))
        invariants = _parse_facts(parsed.get("invariants", []))

        # Infer & correct entity types from declared domain predicate argument signatures
        all_facts = raw_initial_facts + raw_goal_facts + raw_negative_constraints + invariants
        if domain and hasattr(domain, "predicates"):
            for fact in all_facts:
                pred = fact.predicate.lower().rstrip("()")
                if pred in domain.predicates:
                    p_def = domain.predicates[pred]
                    for idx, arg in enumerate(fact.arguments):
                        clean_arg = str(arg).strip().replace(" ", "_").rstrip("()")
                        if idx < len(p_def.arg_types):
                            expected_type = p_def.arg_types[idx]
                            if clean_arg not in entities or entities[clean_arg] not in valid_types:
                                entities[clean_arg] = expected_type
                            elif not domain.type_hierarchy.is_subtype(entities[clean_arg], expected_type):
                                entities[clean_arg] = expected_type

        # Ensure all entity arguments across all facts exist in entities
        for fact in all_facts:
            for arg in fact.arguments:
                clean_arg = str(arg).strip().replace(" ", "_").rstrip("()")
                if clean_arg and clean_arg not in entities:
                    entities[clean_arg] = "block" if is_blocks else (sorted(valid_types)[0] if valid_types else "object")

        # Correct any remaining entity whose type is outside valid_types
        if valid_types:
            for ent_name, ent_type in list(entities.items()):
                if ent_type not in valid_types:
                    entities[ent_name] = "block" if is_blocks else sorted(valid_types)[0]

        # Negative constraints
        negative_constraints = []
        if is_blocks:
            for f in raw_negative_constraints:
                pred = f.predicate.lower().rstrip("()")
                if pred in ("touching", "touch", "move", "moving", "lift", "lifting", "manipulate", "manipulating", "take", "taking"):
                    pred = "holding"
                if pred in ("holding", "on"):
                    negative_constraints.append(FactSchema(predicate=pred, arguments=f.arguments))
        else:
            for f in raw_negative_constraints:
                pred = f.predicate.lower().rstrip("()")
                if not domain or not hasattr(domain, "predicates") or pred in domain.predicates:
                    negative_constraints.append(FactSchema(predicate=pred, arguments=f.arguments))

        # Goal facts
        if is_blocks:
            stacked_in_goal = {f.arguments[0] for f in raw_goal_facts if f.predicate.lower().rstrip("()") == "on" and f.arguments}
            goal_facts = [
                f for f in raw_goal_facts
                if not (f.predicate.lower().rstrip("()") == "holding" and f.arguments and f.arguments[0] in stacked_in_goal)
            ]
        else:
            goal_facts = [
                f for f in raw_goal_facts
                if not domain or not hasattr(domain, "predicates") or f.predicate.lower().rstrip("()") in domain.predicates
            ]

        # Initial facts
        initial_facts: List[FactSchema] = []
        if is_blocks:
            forbidden_held = {f.arguments[0] for f in negative_constraints if f.predicate.lower().rstrip("()") == "holding" and f.arguments}
            prompt_lower = prompt.lower()
            has_held = False
            has_handempty = False

            for f in raw_initial_facts:
                pred = f.predicate.lower().rstrip("()")
                if pred == "holding" and f.arguments:
                    held_arg = f.arguments[0]
                    if held_arg in forbidden_held and "currently holding" not in prompt_lower and "is holding" not in prompt_lower:
                        initial_facts.append(FactSchema(predicate="on_table", arguments=(held_arg,)))
                        initial_facts.append(FactSchema(predicate="clear", arguments=(held_arg,)))
                        continue
                    has_held = True
                elif pred == "handempty":
                    has_handempty = True
                initial_facts.append(FactSchema(predicate=pred, arguments=f.arguments))

            if not has_held and not has_handempty:
                initial_facts.append(FactSchema(predicate="handempty", arguments=()))

            placed_entities = {f.arguments[0] for f in initial_facts if f.predicate.lower() in ("on_table", "on", "holding") and f.arguments}
            for ent in list(entities.keys()):
                if ent not in placed_entities:
                    initial_facts.append(FactSchema(predicate="on_table", arguments=(ent,)))
                    initial_facts.append(FactSchema(predicate="clear", arguments=(ent,)))
        else:
            for f in raw_initial_facts:
                pred = f.predicate.lower().rstrip("()")
                if not domain or not hasattr(domain, "predicates") or pred in domain.predicates:
                    initial_facts.append(f)

        # For non-blocks domains, supplement initial facts / negative constraints from router if model output omitted them
        if not is_blocks and domain and hasattr(self, "router"):
            if not initial_facts or not negative_constraints:
                routed = self.router.route(prompt, explicit_domain_id=domain_id)
                if routed.outcome == RoutingOutcome.SUPPORTED:
                    if not initial_facts and routed.initial_facts:
                        initial_facts.extend(routed.initial_facts)
                        for k, v in routed.entities.items():
                            if k not in entities:
                                entities[k] = v
                    if not negative_constraints and routed.negative_constraints:
                        negative_constraints.extend(routed.negative_constraints)
                        for k, v in routed.entities.items():
                            if k not in entities:
                                entities[k] = v

        def _dedup(facts_list: List[FactSchema]) -> List[FactSchema]:
            seen: Set[Tuple[str, Tuple[str, ...]]] = set()
            out: List[FactSchema] = []
            for item in facts_list:
                k = (item.predicate.lower().rstrip("()"), tuple(item.arguments))
                if k not in seen:
                    seen.add(k)
                    out.append(FactSchema(predicate=k[0], arguments=item.arguments))
            return out

        initial_facts = _dedup(initial_facts)
        goal_facts = _dedup(goal_facts)

        return TaskProposalSchema(
            task_id=str(parsed.get("task_id", "task_formalized")),
            raw_prompt=prompt,
            entities=entities,
            initial_facts=initial_facts,
            goal_facts=goal_facts,
            negative_constraints=negative_constraints,
            invariants=invariants,
            confidence=float(parsed.get("confidence", 0.95)),
        )
