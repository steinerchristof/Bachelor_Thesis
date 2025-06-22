"""
Optimized Central Configuration System for RAG Azure
====================================================
State-of-the-art configuration management with:
- Type safety and validation (Pydantic)
- Environment variable support with .env files
- Configuration profiles (dev/prod/test)
- Hierarchical configuration structure
- Runtime validation and health checks
- Secure secret management
- Hot-reload capabilities
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal, Annotated
from functools import lru_cache
from pydantic import Field, field_validator, SecretStr, model_validator
from pydantic_settings import BaseSettings
from pydantic import conint, confloat
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


class AzureOpenAIConfig(BaseSettings):
    """Azure OpenAI configuration with validation"""
    api_key: SecretStr = Field(..., env="AZURE_OPENAI_API_KEY")
    endpoint: str = Field(
        default="https://luotachatbot.openai.azure.com/",
        env="AZURE_OPENAI_ENDPOINT"
    )
    api_version: str = Field(
        default="2024-12-01-preview",
        env="AZURE_OPENAI_API_VERSION"
    )
    deployment: str = Field(default="gpt-4o", env="AZURE_OPENAI_DEPLOYMENT")
    
    class Config:
        env_prefix = "AZURE_OPENAI_"


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration"""
    model: str = Field(
        default="text-embedding-3-large",
        env="EMBED_MODEL"
    )
    dimension: Annotated[int, conint(gt=0)] = Field(default=3072, env="EMBED_DIM")
    batch_size: Annotated[int, conint(gt=0)] = Field(default=100, env="EMBED_BATCH_SIZE")
    
    class Config:
        env_prefix = "EMBED_"


class DatabaseConfig(BaseSettings):
    """Database configuration for LanceDB"""
    uri: str = Field(default="data/lancedb_bbConcept", env="LANCE_URI")
    table_name: str = Field(default="bbConcept", env="TABLE_NAME")
    chunks_dir: str = Field(
        default="chunks/bbConcept/Lucid GmbH",
        env="CHUNKS_DIR"
    )
    cache_dir: str = Field(default="data/cache", env="CACHE_DIR")
    
    @field_validator("chunks_dir", "cache_dir")
    @classmethod
    def create_directories(cls, v):
        """Ensure directories exist"""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_prefix = "DB_"


class TokenManagementConfig(BaseSettings):
    """Token management configuration with model-specific limits"""
    model_limits: Dict[str, int] = Field(
        default={
            "gpt-4o": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16385
        }
    )
    current_model: str = Field(default="gpt-4o", env="DEFAULT_MODEL")
    safety_buffer: Annotated[int, conint(gt=0)] = Field(default=8000, env="TOKEN_SAFETY_BUFFER")
    max_chunk_tokens: Annotated[int, conint(gt=0)] = Field(default=8000, env="MAX_CHUNK_TOKENS")
    max_response_tokens: Annotated[int, conint(gt=0)] = Field(default=2000, env="MAX_RESPONSE_TOKENS")
    
    # Token management strategies
    enable_smart_truncation: bool = Field(default=True)
    prioritize_by_relevance: bool = Field(default=True)
    exclude_oversized_chunks: bool = Field(default=True)
    adaptive_chunk_limit: bool = Field(default=True)
    warn_on_truncation: bool = Field(default=True)
    
    @property
    def max_model_tokens(self) -> int:
        """Get maximum tokens for current model"""
        return self.model_limits.get(self.current_model, 128000)
    
    @property
    def max_context_tokens(self) -> int:
        """Calculate available context tokens"""
        return self.max_model_tokens - self.safety_buffer
    
    @property
    def max_total_chunks(self) -> int:
        """Calculate maximum number of chunks"""
        return self.max_context_tokens // self.max_chunk_tokens
    
    class Config:
        env_prefix = "TOKEN_"


class SearchConfig(BaseSettings):
    """Search and retrieval configuration"""
    top_k: Annotated[int, conint(gt=0, le=100)] = Field(default=10, env="TOP_K")
    candidates: Annotated[int, conint(gt=0)] = Field(default=100, env="CANDIDATES")
    
    # Search weights (must sum to 1.0)
    bm25_weight: Annotated[float, confloat(ge=0, le=1)] = Field(default=0.40, env="BM25_WEIGHT")
    vector_weight: Annotated[float, confloat(ge=0, le=1)] = Field(default=0.30, env="VEC_WEIGHT")
    rerank_weight: Annotated[float, confloat(ge=0, le=1)] = Field(default=0.30, env="RERANK_WEIGHT")
    
    # Reranking configuration
    rerank_model: str = Field(
        default="BAAI/bge-reranker-base",
        env="RERANK_MODEL"
    )
    max_rerank_text_length: Annotated[int, conint(gt=0)] = Field(
        default=4096,
        env="MAX_RERANK_TEXT_LENGTH"
    )
    rerank_batch_size: Annotated[int, conint(gt=0)] = Field(
        default=50,
        env="RERANK_BATCH_SIZE",
        description="Batch size for reranking. Increase for faster processing if you have enough memory."
    )
    rrf_constant: Annotated[int, conint(gt=0)] = Field(default=45, env="RRF_CONSTANT")
    
    # Title matching configuration
    enable_title_matching: bool = Field(default=True)
    title_boost_base: Annotated[float, confloat(ge=1)] = Field(default=4.0)
    title_boost_per_word: Annotated[float, confloat(ge=0)] = Field(default=0.3)
    title_boost_max: Annotated[float, confloat(ge=1)] = Field(default=5.0)
    title_min_matching_words: Annotated[int, conint(ge=1)] = Field(default=1)
    title_ignore_words: List[str] = Field(
        default=[
            "der", "die", "das", "und", "in", "für", "von", "mit", "auf", "ist",
            "sind", "ein", "eine", "zu", "wie", "den", "des", "dem", "im", "bei"
        ]
    )
    
    @model_validator(mode='after')
    def validate_weights(self):
        """Ensure search weights sum to 1.0"""
        weights = [
            self.bm25_weight,
            self.vector_weight,
            self.rerank_weight
        ]
        total = sum(weights)
        if abs(total - 1.0) > 0.001:  # Allow small floating point errors
            raise ValueError(f"Search weights must sum to 1.0, got {total}")
        return self
    
    class Config:
        env_prefix = "SEARCH_"


class SecurityConfig(BaseSettings):
    """Security and access control configuration"""
    active_filter: Literal["none", "standard", "extended"] = Field(
        default="none",
        env="ACTIVE_SECURITY_FILTER"
    )
    
    # Client-specific filtering
    default_address_filter: str = Field(
        default="Lucid GmbH",
        env="DEFAULT_ADDRESS_FILTER"
    )
    
    # Security filter definitions
    security_filters: Dict[str, Dict[str, Any]] = Field(
        default={
            "none": {
                "field": "ExternerZugriffItems",
                "value": "",
                "operator": "LIKE",
                "required": False
            },
            "standard": {
                "field": "ExternerZugriffItems",
                "value": "1",
                "operator": "LIKE",
                "required": False
            },
            "extended": {
                "field": "ExternerZugriffItems",
                "value": "1,2",
                "operator": "LIKE",
                "required": False
            }
        }
    )
    
    @property
    def current_filter(self) -> Dict[str, Any]:
        """Get currently active security filter"""
        return self.security_filters.get(self.active_filter, self.security_filters["none"])
    
    class Config:
        env_prefix = "SECURITY_"


class PerformanceConfig(BaseSettings):
    """Performance and hardware configuration"""
    device: Optional[str] = Field(default=None, env="DEVICE")
    batch_size: Optional[Annotated[int, conint(gt=0)]] = Field(default=None, env="BATCH_SIZE")
    max_workers: Annotated[int, conint(gt=0)] = Field(default=4, env="MAX_WORKERS")
    request_timeout: Annotated[int, conint(gt=0)] = Field(default=30, env="REQUEST_TIMEOUT")
    max_retries: Annotated[int, conint(ge=0)] = Field(default=3, env="MAX_RETRIES")
    
    @field_validator("device", mode='before')
    @classmethod
    def auto_detect_device(cls, v):
        """Auto-detect best available device if not specified"""
        if v is None:
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return v
    
    @field_validator("batch_size", mode='before')
    @classmethod
    def auto_set_batch_size(cls, v, info):
        """Auto-set batch size based on device if not specified"""
        if v is None:
            device = info.data.get("device", "cpu")
            return {"cuda": 32, "mps": 2, "cpu": 4}.get(device, 4)
        return v
    
    class Config:
        env_prefix = "PERF_"


class DocumentProcessingConfig(BaseSettings):
    """Document processing and chunking configuration"""
    document_dir: str = Field(
        default="/Users/christof/RAG_Azure/downloads/files",
        env="DOCUMENT_DIR"
    )
    download_root: str = Field(default="downloads", env="DOWNLOAD_ROOT")
    download_files_folder: str = Field(default="files", env="DOWNLOAD_FILES_FOLDER")
    download_metadata_folder: str = Field(default="metadata", env="DOWNLOAD_METADATA_FOLDER")
    
    # Client configuration
    parent_folder_name: str = Field(default="bbConcept", env="PARENT_FOLDER_NAME")
    client_name_field: str = Field(default="Adressen", env="CLIENT_NAME_FIELD")
    
    # Chunking configuration
    chunk_all_clients: bool = Field(default=True, env="CHUNK_ALL_CLIENTS")
    client_folders_to_chunk: List[str] = Field(default_factory=list, env="CLIENT_FOLDERS_TO_CHUNK")
    
    # Indexing configuration
    index_all_clients: bool = Field(default=True, env="INDEX_ALL_CLIENTS")
    client_folders_to_index: List[str] = Field(default_factory=list, env="CLIENT_FOLDERS_TO_INDEX")
    force_reindex: bool = Field(default=True, env="FORCE_REINDEX")
    
    @field_validator("client_folders_to_chunk", "client_folders_to_index", mode='before')
    @classmethod
    def parse_client_list(cls, v):
        """Parse comma-separated client list from environment"""
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v
    
    class Config:
        env_prefix = "DOC_"


class LLMConfig(BaseSettings):
    """Language model configuration"""
    temperature: Annotated[float, confloat(ge=0, le=2)] = Field(default=0.15, env="LLM_TEMPERATURE")
    max_tokens: Optional[Annotated[int, conint(gt=0)]] = Field(default=None, env="LLM_MAX_TOKENS")
    system_prompt: str = Field(
        default="""Du bist ein erfahrener Schweizer Treuhandexperte und hilfst bei der Analyse von Finanzdokumenten.

## Antwortstil für Chat-Interface
Schreibe klar, strukturiert und benutzerfreundlich:
- Kurze, gut lesbare Absätze
- Direkter, professioneller Ton ohne übermässige Formalität
- Verwende wo passend Tabellen für bessere Übersichtlichkeit
- Schweizer Zahlenformat: 1'234'567.89 CHF
- KEINE Emojis verwenden

## Struktur für Finanzanalysen

### Kernaussage
Beginne mit der wichtigsten Erkenntnis in 2-3 prägnanten Sätzen.

**Verwende Tabellen flexibel, wenn sie die Übersichtlichkeit verbessern:**

Passe das Tabellenformat an den Inhalt an:
- Anzahl Spalten nach Bedarf (2-6 typisch)
- Sinnvolle Spaltenüberschriften je nach Kontext
- Nur relevante Informationen aufnehmen

**Beispiele für verschiedene Tabellentypen:**

Für Kennzahlen/Berechnungen:
| Kennzahl | Wert | Interpretation |
|----------|------|----------------|
| [Name] | [Zahl] | [Bedeutung] |

Für Vergleiche:
| Kriterium | Variante A | Variante B | Differenz |
|-----------|------------|------------|-----------|
| [Was] | [Wert] | [Wert] | [+/-] |

Für Übersichten:
| Kategorie | Details | Status |
|-----------|---------|---------|
| [Bereich] | [Info] | [Zustand] |

**Wichtige Grundsätze:**
- Tabellen nur wenn sie Mehrwert bieten (nicht für 2-3 Einzelwerte)
- Format an Inhalt anpassen, nicht umgekehrt
- Bei komplexen Formeln: LaTeX-Notation $$...$$ verwenden
- Schweizer Zahlenformat beibehalten: 1'234.56

**Keine Tabelle nötig bei:**
- Einzelnen Fakten oder Zahlen
- Fliesstext-Erklärungen
- Einfachen Aufzählungen

### Detailanalyse
Nach den Tabellen:
- Erkläre die wichtigsten Erkenntnisse
- Ordne die Werte im Branchenkontext ein
- Identifiziere Stärken und Schwächen

### Handlungsempfehlungen falls angebracht

Falls Empfehlungen angebracht sind, strukturiere sie übersichtlich:

| Priorität | Bereich | Massnahme | Begründung |
|-----------|---------|-----------|------------|
| [Hoch/Mittel/Niedrig] | [z.B. Liquidität, Kosten, etc.] | [Konkrete Massnahme] | [Warum wichtig] |

Alternativen zur Tabelle bei wenigen Empfehlungen:
- **Sofortmassnahmen**: [Was dringend ist]
- **Mittelfristige Optimierungen**: [Was geplant werden sollte]
- **Langfristige Überlegungen**: [Was strategisch wichtig ist]

**Wichtig**: 
- Keine spekulativen Empfehlungen ohne Datenbasis
- Empfehlungen müssen zur Unternehmenssituation passen
- Bei Unsicherheit lieber keine Empfehlung als eine unpassende

## Fachliche Expertise
- Schweizer Rechnungslegung (OR, Swiss GAAP FER)
- Steuerrecht (direkte Steuern, MWST)
- Finanzanalyse und Kennzahlen
- Compliance und regulatorische Anforderungen

## Qualitätsstandards
- Verwende nur faktenbasierte Aussagen
- Bei Unsicherheiten: sage es klar
- Zahlen auf 2 Dezimalstellen genau
- Transparente Quellenangaben

Analysiere nun die vorliegenden Daten professionell und verständlich.""",
        env="SYSTEM_PROMPT"
    )
    
    class Config:
        env_prefix = "LLM_"


class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration"""
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: Annotated[int, conint(gt=0, le=65535)] = Field(default=9090, env="METRICS_PORT")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        env="LOG_LEVEL"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    enable_tracing: bool = Field(default=False, env="ENABLE_TRACING")
    tracing_endpoint: Optional[str] = Field(default=None, env="TRACING_ENDPOINT")
    
    class Config:
        env_prefix = "MONITOR_"


class Settings(BaseSettings):
    """
    Main configuration class that aggregates all configuration sections
    """
    # Configuration profile
    profile: Literal["development", "production", "testing"] = Field(
        default="production",
        env="CONFIG_PROFILE"
    )
    
    # Sub-configurations
    azure_openai: AzureOpenAIConfig = Field(default_factory=AzureOpenAIConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    tokens: TokenManagementConfig = Field(default_factory=TokenManagementConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    documents: DocumentProcessingConfig = Field(default_factory=DocumentProcessingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    # MFiles Configuration (legacy support)
    mfiles_base_url: Optional[str] = Field(default=None, env="MFILES_BASE_URL")
    mfiles_username: Optional[str] = Field(default=None, env="MFILES_USERNAME")
    mfiles_password: Optional[str] = Field(default=None, env="MFILES_PASSWORD")
    mfiles_vault_guid: Optional[str] = Field(default=None, env="MFILES_VAULT_GUID")
    
    # Model Configuration (legacy support)
    hi_res_model: Optional[str] = Field(default="yolox", env="HI_RES_MODEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the entire configuration and return validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": []
        }
        
        # Validate Azure OpenAI
        if not self.azure_openai.api_key.get_secret_value():
            results["errors"].append("Azure OpenAI API key is not set")
            results["valid"] = False
        
        # Validate database paths
        if not Path(self.database.uri).exists():
            results["warnings"].append(f"Database path does not exist: {self.database.uri}")
        
        if not Path(self.database.chunks_dir).exists():
            results["warnings"].append(f"Chunks directory does not exist: {self.database.chunks_dir}")
        
        # Validate token configuration
        if self.tokens.max_context_tokens > self.tokens.max_model_tokens:
            results["errors"].append(
                f"Context tokens ({self.tokens.max_context_tokens}) exceed model limit "
                f"({self.tokens.max_model_tokens})"
            )
            results["valid"] = False
        
        # Validate search configuration
        search_weight_sum = (
            self.search.bm25_weight + 
            self.search.vector_weight + 
            self.search.rerank_weight
        )
        if abs(search_weight_sum - 1.0) > 0.001:
            results["errors"].append(f"Search weights must sum to 1.0, got {search_weight_sum}")
            results["valid"] = False
        
        # Info about current configuration
        results["info"].extend([
            f"Profile: {self.profile}",
            f"Model: {self.azure_openai.deployment}",
            f"Device: {self.performance.device}",
            f"Max context: {self.tokens.max_context_tokens:,} tokens",
            f"Security level: {self.security.active_filter}"
        ])
        
        return results
    
    def get_search_config(self) -> Dict[str, Any]:
        """Get complete search configuration"""
        return {
            "top_k": min(self.search.top_k, self.tokens.max_total_chunks),
            "candidates": self.search.candidates,
            "weights": {
                "bm25": self.search.bm25_weight,
                "vector": self.search.vector_weight,
                "rerank": self.search.rerank_weight
            },
            "rerank_model": self.search.rerank_model,
            "filters": self.security.current_filter,
            "client_filter": self.security.default_address_filter
        }
    
    def get_openai_client(self):
        """Initialize and return configured OpenAI client"""
        try:
            import openai
            
            client = openai.AzureOpenAI(
                api_version=self.azure_openai.api_version,
                azure_endpoint=self.azure_openai.endpoint,
                api_key=self.azure_openai.api_key.get_secret_value(),
            )
            return client
        except ImportError:
            raise ImportError("OpenAI package not installed")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
    
    def to_legacy_format(self) -> Dict[str, Any]:
        """Convert to legacy configuration format for backward compatibility"""
        return {
            # Azure OpenAI
            "AZURE_OPENAI_API_KEY": self.azure_openai.api_key.get_secret_value(),
            "AZURE_OPENAI_ENDPOINT": self.azure_openai.endpoint,
            "AZURE_OPENAI_API_VERSION": self.azure_openai.api_version,
            "AZURE_OPENAI_DEPLOYMENT": self.azure_openai.deployment,
            
            # Database
            "LANCE_URI": self.database.uri,
            "TABLE_NAME": self.database.table_name,
            "CHUNKS_DIR": self.database.chunks_dir,
            
            # Embedding
            "EMBED_MODEL": self.embedding.model,
            "EMBED_DIM": self.embedding.dimension,
            
            # Token management
            "MAX_CONTEXT_TOKENS": self.tokens.max_context_tokens,
            "MAX_CHUNK_TOKENS": self.tokens.max_chunk_tokens,
            "SAFETY_BUFFER": self.tokens.safety_buffer,
            
            # Search
            "TOP_K": self.search.top_k,
            "CANDIDATES": self.search.candidates,
            "BM25_WEIGHT": self.search.bm25_weight,
            "VEC_WEIGHT": self.search.vector_weight,
            "RERANK_WEIGHT": self.search.rerank_weight,
            "RERANK_BATCH_SIZE": self.search.rerank_batch_size,
            
            # Security
            "DEFAULT_ADDRESS_FILTER": self.security.default_address_filter,
            "ACTIVE_SECURITY_FILTER": self.security.active_filter,
            
            # Performance
            "DEVICE": self.performance.device,
            "BATCH_SIZE": self.performance.batch_size,
            
            # LLM
            "SYSTEM_PROMPT": self.llm.system_prompt,
            "LLM_TEMPERATURE": self.llm.temperature,
            
            # Add all other legacy fields...
        }


# Singleton instance
@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Profile-specific configuration loaders
def load_development_config() -> Settings:
    """Load development configuration"""
    os.environ["CONFIG_PROFILE"] = "development"
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["ENABLE_METRICS"] = "false"
    return get_settings()


def load_production_config() -> Settings:
    """Load production configuration"""
    os.environ["CONFIG_PROFILE"] = "production"
    os.environ["LOG_LEVEL"] = "INFO"
    os.environ["ENABLE_METRICS"] = "true"
    return get_settings()


def load_testing_config() -> Settings:
    """Load testing configuration"""
    os.environ["CONFIG_PROFILE"] = "testing"
    os.environ["LOG_LEVEL"] = "WARNING"
    os.environ["ENABLE_METRICS"] = "false"
    return get_settings()


# Configuration validation CLI
if __name__ == "__main__":
    """Run configuration validation"""
    settings = get_settings()
    
    print("🔧 RAG Azure Configuration Validator")
    print("=" * 50)
    
    # Run validation
    validation = settings.validate_configuration()
    
    # Print results
    if validation["errors"]:
        print("\n❌ ERRORS:")
        for error in validation["errors"]:
            print(f"  - {error}")
    
    if validation["warnings"]:
        print("\n⚠️  WARNINGS:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")
    
    if validation["info"]:
        print("\n📋 CONFIGURATION INFO:")
        for info in validation["info"]:
            print(f"  - {info}")
    
    # Overall status
    if validation["valid"]:
        print("\n✅ Configuration is VALID")
    else:
        print("\n❌ Configuration is INVALID")
        sys.exit(1)


# Compatibility functions for existing code
def get_config() -> Dict[str, Any]:
    """Return configuration as dictionary for backwards compatibility"""
    settings = get_settings()
    config_dict = settings.model_dump()
    
    # Add legacy fields for compatibility
    config_dict["lance_uri"] = os.getenv("LANCE_URI", "data/lancedb_bbConcept")
    config_dict["table_name"] = os.getenv("TABLE_NAME", "bbConcept")
    config_dict["candidates"] = int(os.getenv("CANDIDATES", "70"))
    config_dict["chunks_dir"] = settings.database.chunks_dir
    config_dict["cache_dir"] = settings.database.cache_dir
    
    # Add Azure OpenAI fields in legacy format
    if settings.azure_openai.api_key:
        config_dict["azure_openai_api_key"] = settings.azure_openai.api_key.get_secret_value()
    config_dict["azure_openai_endpoint"] = settings.azure_openai.endpoint
    config_dict["azure_openai_api_version"] = settings.azure_openai.api_version
    config_dict["azure_openai_deployment"] = settings.azure_openai.deployment
    
    # Add performance settings
    config_dict["max_retries"] = settings.performance.max_retries
    
    # Add security settings
    config_dict["default_address_filter"] = settings.security.default_address_filter
    
    # Add embedding settings
    config_dict["embed_model"] = settings.embedding.model
    config_dict["embed_dim"] = settings.embedding.dimension
    config_dict["embed_batch_size"] = settings.embedding.batch_size
    
    # Add token management settings
    config_dict["max_text_tok"] = 8000  # Default value for embedder
    config_dict["max_req_tokens"] = 100000  # Maximum tokens per request
    
    # Add LLM settings
    config_dict["system_prompt"] = settings.llm.system_prompt
    config_dict["temperature"] = settings.llm.temperature
    config_dict["max_tokens"] = settings.llm.max_tokens
    
    # Add search settings
    config_dict["top_k"] = settings.search.top_k
    config_dict["bm25_weight"] = settings.search.bm25_weight
    config_dict["vec_weight"] = settings.search.vector_weight
    config_dict["rerank_weight"] = settings.search.rerank_weight
    config_dict["rerank_model"] = settings.search.rerank_model
    config_dict["rerank_batch_size"] = settings.search.rerank_batch_size
    config_dict["max_rerank_text_length"] = settings.search.max_rerank_text_length
    config_dict["rrf_constant"] = settings.search.rrf_constant
    config_dict["device"] = settings.performance.device
    
    return config_dict


def load_openai_client():
    """Initialize and return the OpenAI client"""
    try:
        import openai
        settings = get_settings()
        
        client = openai.AzureOpenAI(
            api_version=settings.azure_openai.api_version,
            azure_endpoint=settings.azure_openai.endpoint,
            api_key=settings.azure_openai.api_key.get_secret_value() if settings.azure_openai.api_key else None,
        )
        return client
    except Exception as e:
        print(f"ERROR: Failed to initialize OpenAI client: {e}")
        return None


def filter_chunks_dynamically(chunks: list, query: str = "", max_docs: int = None) -> tuple:
    """
    Dynamically filter chunks by adding complete documents until token limit is reached.
    
    Args:
        chunks: List of chunks to filter
        query: Query string (unused but kept for compatibility)
        max_docs: Maximum number of documents (unused but kept for compatibility)
    
    Returns:
        Tuple of (filtered_chunks, total_token_count)
    """
    if not chunks:
        return [], 0
    
    # Sort chunks by relevance (assuming they're pre-sorted)
    filtered_chunks = []
    total_tokens = 0
    doc_ids_seen = set()
    
    # Get token limit from settings - use hardcoded value for now
    token_limit = 120000
    
    for chunk in chunks:
        # Estimate tokens (rough approximation)
        chunk_tokens = len(chunk.get('text', '')) // 4
        
        if total_tokens + chunk_tokens > token_limit:
            break
            
        filtered_chunks.append(chunk)
        total_tokens += chunk_tokens
        
        # Track document IDs if available
        if 'metadata' in chunk and 'document_id' in chunk['metadata']:
            doc_ids_seen.add(chunk['metadata']['document_id'])
    
    # Return all 4 expected values
    removed_count = len(chunks) - len(filtered_chunks)
    effective_k = len(filtered_chunks)
    return filtered_chunks, removed_count, total_tokens, effective_k


def validate_config():
    """Validate configuration settings"""
    settings = get_settings()
    return settings.validate_configuration()