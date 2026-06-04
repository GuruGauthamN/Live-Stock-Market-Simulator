from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = Field(default="stock-predicting-agent", alias="APP_NAME")
    data_dir: str = Field(default="data", alias="DATA_DIR")
    artifacts_dir: str = Field(default="artifacts", alias="ARTIFACTS_DIR")
    symbols: str = Field(default="AAPL,MSFT,GOOGL", alias="SYMBOLS")
    stream_interval_seconds: int = Field(default=1, alias="STREAM_INTERVAL_SECONDS")
    news_interval_seconds: int = Field(default=20, alias="NEWS_INTERVAL_SECONDS")
    news_provider: str = Field(default="mock", alias="NEWS_PROVIDER")
    news_api_key: str = Field(default="", alias="NEWS_API_KEY")
    news_language: str = Field(default="en", alias="NEWS_LANGUAGE")
    model_min_bars: int = Field(default=200, alias="MODEL_MIN_BARS")
    prediction_horizon_bars: int = Field(default=5, alias="PREDICTION_HORIZON_BARS")
    signal_threshold: float = Field(default=0.55, alias="SIGNAL_THRESHOLD")
    invest_news_floor: float = Field(default=-0.1, alias="INVEST_NEWS_FLOOR")
    auto_execute_decisions: bool = Field(default=True, alias="AUTO_EXECUTE_DECISIONS")
    decision_trade_qty: int = Field(default=1, alias="DECISION_TRADE_QTY")
    llm_enabled: bool = Field(default=False, alias="LLM_ENABLED")
    llm_auto_run: bool = Field(default=False, alias="LLM_AUTO_RUN")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_base_url: str = Field(default="https://api.openai.com/v1", alias="LLM_BASE_URL")
    llm_model: str = Field(default="gpt-4.1-mini", alias="LLM_MODEL")
    llm_decision_interval_seconds: int = Field(
        default=300, alias="LLM_DECISION_INTERVAL_SECONDS"
    )
    llm_min_confidence: float = Field(default=0.6, alias="LLM_MIN_CONFIDENCE")
    max_position_value: float = Field(default=5000, alias="MAX_POSITION_VALUE")
    max_daily_loss: float = Field(default=500, alias="MAX_DAILY_LOSS")
    starting_cash: float = Field(default=100000, alias="STARTING_CASH")

    @property
    def symbol_list(self) -> list[str]:
        return [s.strip().upper() for s in self.symbols.split(",") if s.strip()]

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def artifacts_path(self) -> Path:
        return Path(self.artifacts_dir)

    @property
    def market_data_file(self) -> Path:
        return self.data_path / "market_data.csv"

    @property
    def news_data_file(self) -> Path:
        return self.data_path / "news_data.csv"

    @property
    def decision_file(self) -> Path:
        return self.data_path / "decision_history.csv"

    @property
    def model_file(self) -> Path:
        return self.artifacts_path / "signal_model.joblib"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_path.mkdir(parents=True, exist_ok=True)
    settings.artifacts_path.mkdir(parents=True, exist_ok=True)
    return settings
