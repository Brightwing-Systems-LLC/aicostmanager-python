# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-02-19

### Removed — BREAKING CHANGES

These endpoints have been removed from the server in favor of MCP tools.
Analytics, reports, and limit event queries are now available exclusively
through the MCP server at `mcp.aicostmanager.com`.

- **Analytics methods** (12 methods removed from both sync and async clients):
  - `analytics_costs_daily()`, `analytics_costs_monthly()`
  - `analytics_costs_snapshots()`, `analytics_costs_trends()`
  - `analytics_costs_peak_usage()`
  - `analytics_customers_costs()`, `analytics_customers_tokens()`
  - `analytics_services_ranking()`, `analytics_services_efficiency()`,
    `analytics_services_usage()`
  - `analytics_vendors_comparison()`, `analytics_vendors_usage()`
- **Reports methods** (3 methods removed):
  - `list_reports()`, `get_report()`, `download_report()`
- **Limit events method** (1 method removed):
  - `list_limit_events()`
- **Export schedule/job methods** (8 methods removed — already broken since
  server-side removal in 0.1.x):
  - `create_export_schedule()`, `list_export_schedules()`,
    `get_export_schedule()`, `update_export_schedule()`,
    `delete_export_schedule()`
  - `list_export_jobs()`, `get_export_job()`, `trigger_export_job()`
- **Deleted model modules**:
  - `models/analytics.py` — all analytics filter/response schemas
  - `models/reports.py` — `GeneratedReportOut`
  - `models/schedules.py` — all export schedule/job schemas

### Migration Guide

If you use any of the removed methods, connect to the MCP server instead.
The MCP server provides equivalent (and enhanced) functionality:

| Removed SDK method | MCP tool replacement |
| --- | --- |
| `analytics_costs_daily()` | `get_analytics_costs_daily` |
| `analytics_costs_snapshots()` | `get_analytics_costs_snapshots` |
| `list_reports()` | `get_reports` |
| `download_report()` | `get_report_download_url` |
| `list_limit_events()` | `get_limit_events` |
| `create_export_schedule()` | Use `request_on_demand_export` or web UI |

See <https://aicostmanager.com/mcp-server/> for the full MCP tool reference.

## [0.1.41] - 2025-10-07

### Fixed
- **FireworksWrapper Model Extraction**: Fixed `FireworksWrapper` to correctly extract model names from `fireworks.LLM` clients:
  - Added support for clients that store the model as an instance attribute (e.g., `fireworks.LLM`)
  - Maintains backward compatibility with legacy `fireworks.client.Fireworks` clients
  - Prevents "unknown-model" service keys when model is not passed in API call parameters
  - Service keys now correctly show `fireworks-ai::{model_name}` instead of `fireworks-ai::unknown-model`

## [0.1.40] - 2025-10-07

### Added
- **Fireworks AI LLM Wrapper**: New `FireworksWrapper` class for tracking Fireworks AI API usage:
  - Full support for Fireworks client completions API with `completions.create()` method
  - Automatic vendor attribution as `fireworks-ai` in service keys (format: `fireworks-ai::[model]`)
  - Support for both streaming and non-streaming completions
  - Comprehensive test coverage including mock tests and real API integration tests
  - Added `fireworks-ai` package to test dependencies
  - Updated documentation with Fireworks usage examples

## [0.1.39] - 2025-09-26

### Added
- **Custom Services API Support**: Complete implementation of custom service configuration management:
  - New `CustomServiceIn`, `CustomServiceOut`, `CustomServiceSummaryOut` models
  - Advanced cost unit configuration with `CustomCostUnitIn`/`CustomCostUnitOut`
  - Flexible mapping system supporting `Mapping`, `MappingGroup`, `Condition`, and `CostUnitSwitch`
  - Exclusions support for filtering events
  - Full CRUD operations: `list_custom_services()`, `create_custom_service()`, `get_custom_service()`, `update_custom_service()`, `delete_custom_service()`
  - Comprehensive validation and type safety

- **Batch Tracking Support**: Efficient bulk usage tracking with `track_batch()` and `track_batch_async()`:
  - Send multiple tracking records in a single HTTP request
  - Automatic batch size limits (1000 records max)
  - Configurable anonymization and error handling
  - Improved performance for high-volume tracking scenarios
  - Full async support with `track_batch_async()`

### Changed
- **API Parameter Updates**: Removed `api_id` parameter from all track endpoint calls:
  - Updated `track()` method calls throughout codebase to use `service_key` instead of `api_id`
  - Fixed documentation examples in README.md, docs/fastapi.md, docs/streamlit.md, docs/django.md
  - Ensured HTTP requests to `/track` endpoint no longer include `api_id` fields
  - Updated all test cases to use correct parameter signatures

### Fixed
- **E2E Test Reliability**: Fixed failing end-to-end tests for usage limits:
  - Corrected limit amounts to ensure triggering (extremely small values like 0.0000000001)
  - Fixed persistent queue test expectations to account for asynchronous processing
  - Improved test resilience for server-side timing variations
  - Added proper validation of both exception handling and triggered limits state

### Enhanced
- **Model Type Safety**: Improved Pydantic models with comprehensive field descriptions and validation
- **Error Handling**: Better handling of API responses and edge cases in custom services
- **Documentation**: Updated examples to reflect current API usage patterns

## [0.1.38] - 2025-09-25
### Changed
- **API Request Format Updates**: Updated test suite to align with server-side API changes:
  - Removed `api_id` field from direct HTTP request bodies as the API no longer accepts this parameter
  - Updated error handling expectations to match current API behavior where invalid service keys return `service_key_unknown` status instead of errors
  - Modified payload validation expectations where invalid payloads are now queued (`status: 'queued'`) instead of rejected with errors
  - Updated test assertions to properly validate the new response formats for various error conditions

### Fixed
- **Test Compatibility**: Fixed failing `test_deliver_now_multiple_events_with_errors` test to work with the updated server API response formats and validation rules

## [0.1.37] - 2025-09-19
### Removed
- **User-Based Limit Logic**: Completely removed backward compatibility for user-specific limits. The following components no longer support user-based filtering:
  - `TriggeredLimit` class no longer has `user` field
  - `TriggeredLimitPayload` model no longer has `user` field
  - `TriggeredLimitsCache` no longer stores or retrieves user information
  - Removed user filtering logic from triggered limits processing
  - Removed `test_triggered_limit_user_filter` test function

### Enhanced
- **Simplified Triggered Limits**: Streamlined triggered limits functionality by removing user-based complexity:
  - Cleaner filtering logic that only considers service key and customer key matching
  - Reduced cache complexity with focus on core data storage
  - Simplified model definitions without legacy user fields
  - Improved performance by eliminating unnecessary user comparison operations

### Changed
- **API Compatibility**: Updated internal models to no longer expect or handle user fields in triggered limit responses, aligning with the current API specification that no longer supports user-based limits

## [0.1.36] - 2025-09-18
### Changed
- **API Parameter Rename**: Renamed `client_customer_key` to `customer_key` throughout the entire codebase for consistency and simplicity. This affects all models, methods, and documentation. The `set_client_customer_key()` method is now `set_customer_key()`.

## [0.1.35] - 2025-09-16
### Changed
- **Default Error Handling**: Changed `AICM_RAISE_ON_ERROR` default from `true` to `false`. By default, tracking failures now log errors and continue instead of raising exceptions. Set `AICM_RAISE_ON_ERROR=true` to restore the previous behavior.
- **API Model Updates**: Updated cost event models to align with latest API:
  - `CostEvent` renamed to `CostEventItem`
  - `vendor_id` field renamed to `provider_id`
  - `service_id` field renamed to `service_key`
  - Enhanced type flexibility for cost fields to support both numeric and string values
  - Updated `ErrorResponse` schema: `error`/`message` fields renamed to `detail`/`code` for API consistency
- **API Parameter Updates**: Renamed `api_id` parameter to `service_key` across all Tracker methods for consistency with the updated API
- **Environment Variable Cleanup**: Removed legacy `AICM_DELIVERY_LOG_BODIES` environment variable in favor of the standardized `AICM_LOG_BODIES`

### Added
- **Vendor-API Mapping**: Added `_get_vendor_api_mapping()` helper method to Tracker class for extracting vendor and API information from service keys
- **Service Key Building**: Added `_build_final_service_key()` method for constructing proper service keys from API responses
- **Parameter Resolution**: Added `_resolve_tracking_params()` helper for intelligent fallback handling of client metadata
- **Response Metadata Attachment**: Added `_attach_tracking_metadata()` method to safely attach tracking results to LLM response objects
- **Enhanced Type Support**: Improved cost field handling to support flexible numeric/string types in API responses
- **Code Organization**: Added comprehensive section comments and method organization within Tracker class
- **Expanded Model Library**: Added comprehensive new model classes for tracking, analytics, webhooks, and custom services:
  - `TrackRequest`, `TrackResponse`, `TrackResult`, `TrackedRecord` for tracking operations
  - `TriggeredLimitPayload` for encrypted limit data
  - Analytics models: `CustomerBreakdownSchema`, `TrendsResponseSchema`, `SnapshotsResponseSchema`
  - Webhook models: `WebhookEndpointCreate`, `WebhookEndpointOut`, `WebhookEndpointsResponse`
  - Custom service models: `CustomServiceIn`, `CustomServiceOut`, `CustomCostUnitIn`
  - Export scheduling models: `ExportScheduleCreate`, `ExportScheduleOut`, `ExportJobsResponse`

### Fixed
- **Test Compatibility**: Updated mock tracker classes in test files to include `ini_manager` attribute required by LLM wrapper initialization, fixing AttributeError failures in:
  - `tests/test_llm_wrappers.py`
  - `tests/test_magicmock_compatibility.py`
  - `tests/test_wrapper_delivery_type.py`
- **API Response Handling**: Updated CostQueryManager to work with new `CostEventItem` model structure and return type annotations
- **Service Key Corrections**: Fixed service key handling and mapping issues across various components

## [0.1.34] - 2025-01-27
### Added
- Optional pre-inference limit enforcement in LLM wrappers controlled by
  `AICM_ENABLE_INFERENCE_BLOCKING_LIMITS`.

## [0.1.33] - 2025-01-27
### Added
- **Automatic Fallback Values**: `track()` and `track_async()` methods now automatically use instance-level `customer_key` and `context` values when no explicit parameters are provided, eliminating the need to pass these parameters on every call

### Enhanced
- **Parameter Override Support**: Explicit method parameters take precedence over instance-level values when provided
- **Developer Experience**: Significantly simplified metadata management for users who want to set customer context once per tracker instance rather than per tracking call

## [0.1.32] - 2025-01-27
### Added
- **Tracker Instance-Level Metadata**: Added `set_customer_key()` and `set_context()` methods to the base `Tracker` class for storing customer key and context information at the instance level
- **Enhanced Tracker API**: Base `Tracker` class now supports instance-level configuration of `customer_key` and `context` values, providing consistency with LLM wrapper classes

### Enhanced
- **API Consistency**: Improved API consistency across Tracker and wrapper classes by adding instance-level setters for metadata tracking
- **Developer Experience**: Simplified metadata management for users who want to set customer context once per tracker instance rather than per tracking call

## [0.1.31] - 2025-09-03
### Added
- **Gemini 2.0 Flash Support**: Added comprehensive support for `gemini-2.0-flash` model across all Gemini tests and usage tracking
- **Enhanced Gemini Usage Fields**: Added support for advanced Gemini usage fields including:
  - `cachedContentTokenCount` - tokens served from cache
  - `toolUsePromptTokenCount` - tokens used for tool/function calling
  - `thoughtsTokenCount` - tokens used for model thinking/reasoning
  - `promptTokensDetails` - per-modality token breakdown for prompts
  - `candidatesTokensDetails` - per-modality token breakdown for responses
  - `cacheTokensDetails` - detailed cache token usage
- **Modality Token Count Serialization**: Added handling for `ModalityTokenCount` objects in Gemini API responses with proper JSON serialization

### Enhanced
- **Gemini Usage Normalization**: Improved `_normalize_gemini_usage()` function to handle both camelCase and snake_case field variants
- **Response ID Extraction**: Enhanced response ID detection to try multiple field names (`id`, `response_id`, `responseId`) for better Google API correlation
- **Streaming Usage Tracking**: Updated streaming usage extraction to use centralized normalization for consistency
- **Test Coverage**: Expanded Gemini test suite from 10 to 19 tests covering both `gemini-2.5-flash` and `gemini-2.0-flash` models

### Fixed
- **JSON Serialization Error**: Resolved `TypeError: Object of type ModalityTokenCount is not JSON serializable` in Gemini usage tracking
- **Usage Field Handling**: Fixed handling of complex Gemini API response objects in usage extraction pipeline

### Documentation
- **Gemini Usage Guide**: Added comprehensive documentation (`docs/gemini.md`) covering all supported usage fields, extraction methods, and integration details
- **API Response Handling**: Documented handling of both streaming and non-streaming Gemini API responses

## [0.1.30] - 2025-09-02
### Added
- User-specific triggered limits support for enhanced limit management
- Configurable error handling for immediate delivery with `on_failure` parameter
- Dynamic context and client customer key support for LLM wrappers
- Service key-based triggered limits testing and validation

### Enhanced
- **Retry Logic**: Added `reraise=True` parameter to delivery retry mechanism for better error propagation
- **Triggered Limits**: Enhanced triggered limits cache to handle user-specific limits alongside service-based limits
- **Error Handling**: Immediate delivery now supports configurable failure handling (raise, log, or ignore)
- **LLM Wrappers**: Expanded support for dynamic context updates between API calls
- **Testing**: Comprehensive test coverage for service key-based triggered limits and delivery type forwarding

### Removed
- **Legacy Endpoints**: Removed deprecated track-usage endpoint and related models for cleaner API surface
- **Outdated Configuration**: Cleaned up references to `AICM_DELIVERY_ON_FULL` environment variable

### Fixed
- **Delivery Configuration**: Improved delivery type forwarding and configuration precedence
- **Test Coverage**: Enhanced wrapper delivery type forwarding tests

## [0.1.29] - 2025-01-27
### Added
- LLM wrappers now accept ``customer_key`` and ``context`` at
  initialization and allow updating them between inferences for tracking.
- Expanded real wrapper tests to verify per-inference updates across
  streaming and non-streaming calls.

## [0.1.28] - 2025-01-27
### Fixed
- **Critical MagicMock Compatibility**: Resolved infinite recursion issue in usage tracking when using MagicMock objects in tests
- **Test Stability**: Fixed hanging test `test_openai_chat_wrapper_magicmock_non_streaming` that was causing test timeouts
- **Usage Serialization**: Enhanced `_to_serializable_dict()` function to safely handle Mock/MagicMock objects without triggering dynamic attribute creation

### Enhanced
- **Mock Object Detection**: Added early detection of Mock, MagicMock, and AsyncMock objects in usage extraction pipeline
- **Test Safety**: Improved `_is_unsafe_object()` function to classify mock objects as unsafe for JSON serialization
- **Developer Experience**: Better testing compatibility for developers using unittest.mock in their LLM wrapper tests

### Added
- **Comprehensive Mock Testing**: Enhanced test suite with 7 MagicMock compatibility tests covering attribute access, streaming, deep nesting, and callable objects
- **Safe Mock Handling**: Mock objects now return empty usage dictionaries instead of causing infinite loops during serialization

## [0.1.27] - 2025-01-27
### Fixed
- **Critical Test Compatibility**: Fixed missing `log_bodies` attribute in `ImmediateDelivery` class that was causing 35 test failures with `AttributeError: 'ImmediateDelivery' object has no attribute 'log_bodies'`
- **Delivery Consistency**: Both `ImmediateDelivery` and `PersistentDelivery` now consistently support the `log_bodies` parameter for debugging and troubleshooting
- **Factory Function**: Updated `create_delivery` factory to properly handle `log_bodies` parameter for immediate delivery, reading from kwargs, environment variables (`AICM_LOG_BODIES` or `AICM_DELIVERY_LOG_BODIES`), with fallback to `False`

### Enhanced
- **Parameter Compatibility**: `ImmediateDelivery` constructor now accepts `log_bodies: bool = False` parameter for consistency with `PersistentDelivery`
- **Environment Variable Support**: Both delivery types now respect `AICM_LOG_BODIES` environment variable for enabling request/response body logging

## [0.1.26] - 2025-01-27
### Enhanced
- **Delivery Classes Simplified Initialization**: Major improvement to both `PersistentDelivery` and `ImmediateDelivery` constructors making them much easier to use with intelligent defaults
- **Configuration Management**: Enhanced INI file integration - both delivery classes now automatically read configuration overrides from `AICM.INI` including `AICM_API_BASE`, `AICM_DB_PATH`, timeouts, and logging settings
- **HTTP Client Error Handling**: Improved error handling in delivery base class to detect and handle closed HTTP clients gracefully, preventing unnecessary retry attempts
- **FastAPI Integration**: Streamlined FastAPI documentation focusing on modern lifespan context manager patterns with simplified tracker initialization

### Added
- **Intelligent Defaults**: Both `PersistentDelivery()` and `ImmediateDelivery()` can now be instantiated with zero parameters - automatically use `AICM_API_KEY` from environment and read configuration overrides from INI files
- **Consistent API**: Both delivery classes now have identical initialization patterns for easy switching between delivery types
- **Configuration Priority System**: Clear precedence order - environment variables (highest), INI file settings, hardcoded defaults (fallback)
- **One-line Setup**: Simplified from complex configuration to `DeliveryClass()` + `Tracker(delivery=delivery)` pattern for both delivery types

### Changed
- **Breaking**: FastAPI documentation now focuses only on lifespan context manager (removed deprecated startup/shutdown events since no one is using them yet)
- **Simplified Examples**: Updated all documentation examples to use new simplified initialization patterns
- **Better Error Messages**: HTTP client closure errors now provide clearer messages and don't attempt futile retries

### Fixed
- **Race Condition**: Resolved HTTP client closure issue that occurred when tracker was closed while async operations were still pending
- **Documentation Consistency**: Updated HeyGen, tracker, and persistent delivery documentation to use new simplified patterns

## [0.1.24] - 2025-01-27
### Enhanced
- **HeyGen Integration**: Comprehensive step-by-step documentation (`docs/heygen.md`) covering complete API integration workflow from session retrieval to cost tracking
- **HeyGen Testing**: Improved test coverage with separate session sets for immediate vs persistent delivery methods to prevent duplicate tracking
- **Deepgram Support**: Enhanced documentation and testing for websocket transcription and streaming text-to-speech services
- **Persistent Delivery**: Improved queue management leveraging automatic context manager flushing, reducing manual drainage wait logic while maintaining reliability

### Added
- Complete HeyGen API integration examples with both immediate and persistent delivery methods
- Production-ready HeyGen sync functions suitable for cron jobs and scheduled tasks
- Enhanced error handling and troubleshooting guidance for HeyGen API integration
- Improved test isolation ensuring each session is tracked only once across delivery types

### Fixed
- Simplified persistent queue drainage by better utilizing the automatic `stop()`/`close()` behavior in context managers
- Reduced unnecessary manual timeout logic while maintaining test reliability
- Updated import organization and code formatting consistency

## [0.1.23] - 2025-01-27
### Fixed
- `PersistentDelivery` logger initialization issue that caused `AttributeError: 'PersistentDelivery' object has no attribute 'logger'` when accessing the logger during database setup before parent class initialization
- Initialize logger early in `PersistentDelivery.__init__()` to ensure it's available for database operations and warning messages about failed queue items

### Added
- `track_llm_usage` and `track_llm_usage_async` methods for automatic usage
  extraction from LLM responses, with streaming helpers.
- `AICM_LIMITS_ENABLED` configuration to toggle triggered limit enforcement in
  delivery components.


## [0.1.18] - 2025-08-11
### Added

- REST API tracking guide (`docs/rest.md`) and example script `examples/heygen_sync.py` showing batch/manual tracking with `Tracker`
- Per-call context support in REST wrappers via `set_customer_key()` and `set_context()`

### Enhanced
- `UniversalExtractor`: base_url autodetection, streaming `final_fields` support, UUID generation for missing `response_id`, and broader JSON serialization (httpx.URL/urllib `ParseResult`, Pydantic `model_dump`, objects with `dict`/`__dict__`)
- `RestCostManager` and `AsyncRestCostManager`: pre-request triggered limit checks, delivery batching controls (`delivery_mode`, `on_full`, batch size/interval), and context manager helpers
- Documentation refreshed with REST wrapper usage and FastAPI tracker lifecycle guidance

### Fixed
- More defensive extraction error handling so failures never block delivery
- Minor documentation link/formatting consistency

## [0.1.15] - 2025-01-07
### Enhanced
- Documentation updated to be uv-first across the project (installation and dev workflows)
- README now recommends `uv pip install` / `uv add` for installation and adds queue health examples
- Usage guide now shows uv-first install instructions
- Build & Deploy guide updated to use `uv pip` for dev and release verification commands
- Testing guide aligned on `uv venv` and `uv pip` for dependency installs

### Added
- New `docs/config.md` covering configuration settings and defaults
- New `docs/tracker.md` with comprehensive manual usage tracking guide, validation, async init, and FastAPI lifecycle

### Fixed
- Changelog wording referencing pip updated to `uv pip install` where appropriate

## [0.1.14] - 2025-01-07
### Added
- **New `Tracker` class** for manual usage tracking with configurable validation schema
- `UsageValidationError` exception for robust type validation of tracked usage data
- `TypeValidator` utility for validating values against string-specified Python type hints
- Async factory method `Tracker.create_async()` for non-blocking tracker initialization in async applications
- `Tracker.close()` method for graceful shutdown of background delivery workers
- Enhanced delivery queue with overflow metrics and configurable discard callbacks

### Enhanced
- **Manual usage tracking capabilities** with schema-based validation for custom usage scenarios
- Delivery system performance improvements with persistent event loop reuse in async mode
- FastAPI integration examples with proper lifecycle management for tracker instances
- Comprehensive test coverage for new tracker functionality and validation scenarios
- Queue overflow handling with three modes: `block`, `raise`, or `backpressure` (with discard metrics)

### Fixed
- **Triggered limits retrieval** now properly fetches from API when local cache is missing or corrupted
- Improved reliability of end-to-end usage limit enforcement tests
- Better event loop management in delivery worker threads for async operations
- Enhanced config manager robustness when INI files lack required encrypted payloads

### Technical Improvements
- Added `on_discard` callback support for monitoring dropped payloads in delivery queues
- Persistent event loop management in delivery worker for better async performance
- Enhanced triggered limits initialization with automatic API fallback when cache is empty
- Comprehensive validation framework supporting complex Python type hints including `Optional`, `Union`, `List[T]`, and `Dict[K,V]`

## [0.1.13] - 2025-01-06
### Enhanced
- Updated build and deployment documentation with comprehensive step-by-step release process
- Added detailed prerequisites, troubleshooting, and verification steps for releases
- Improved documentation for `bump-my-version` integration and GitHub Actions workflow
- Added quick reference commands for common release tasks

## [0.1.12] - 2025-01-05
### Fixed
- **CRITICAL**: Fixed empty package builds by correcting setuptools package discovery configuration
- Packages built with previous versions (0.1.10 and earlier) contained only metadata without actual Python source code
- This resolves issues where `uv add aicostmanager` and `uv pip install aicostmanager` resulted in empty package installations
- Updated `pyproject.toml` to properly include source files in wheel distributions

## [0.1.11] - 2025-08-04
### Fixed
- Fixed the version number in the __init__


## [0.1.10] - 2025-08-04
### Added
- Custom client metadata support with `customer_key` and `context` parameters
- New `set_customer_key()` and `set_context()` methods for all cost manager classes
- Automatic augmentation of usage payloads with client metadata for tracking and identification
- Client metadata support across all cost manager types: `CostManager`, `AsyncCostManager`, `RestCostManager`, and `AsyncRestCostManager`

### Enhanced
- Updated project dependencies to latest versions for improved compatibility and security
- Enhanced test coverage with comprehensive client metadata validation across all LLM providers
- Improved payload tracking with consistent metadata injection across sync and async operations

### Technical Improvements
- Unified `_augment_payload()` method implementation across all cost manager classes
- Comprehensive test suite covering client metadata functionality for OpenAI, Anthropic, Bedrock, Gemini, and OpenAI-compatible providers
- Enhanced delivery verification tests to validate client metadata presence in usage events

## [0.1.9] - 2025-01-30
### Fixed
- Fixed triggered limits initialization reliability during `CostManagerClient` instantiation
- Resolved test execution order dependency in `test_usage_limit_end_to_end` that caused intermittent failures
- Fixed global delivery state interference between test suites by adding proper test isolation
- Prevented spurious "ini" file creation in project root directory during mock testing
- Enhanced environment variable cleanup in client configuration tests

### Enhanced
- Improved triggered limits lifecycle management with consistent ETag-based fetching logic
- Enhanced config manager alignment with client initialization for better state consistency
- Added robust test isolation with `clean_delivery` fixture to prevent cross-test pollution
- Improved mock test safety by preventing file system operations with literal "ini" paths

### Added
- `RestCostManager` class for tracking costs of generic REST API calls using `requests.Session`
- `AsyncRestCostManager` class for tracking costs of async REST API calls using `httpx.AsyncClient`
- Universal REST API cost tracking capabilities for non-LLM services and custom APIs
- Support for tracking HTTP method calls (GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS) with automatic usage extraction

### Technical Improvements
- All 81 core tests now pass reliably regardless of execution order
- Usage limit enforcement works consistently across different test scenarios
- Better separation between mock and real client behavior in test environment
- Enhanced triggered limits state management in INI file persistence layer

## [0.1.8] - 2025-01-30
### Fixed
- Fixed syntax error in `test_async_client.py` where elif statement had incorrect indentation
- Fixed config manager test mock functions to properly accept `etag` parameter
- Added missing `configs_etag` attribute and property to `AsyncCostManagerClient` class
- Resolved ETag payload validation issues in comprehensive testing scenarios

### Added
- Comprehensive ETag functionality test suite (`test_real_endpoints_etag.py`) with 13 real API tests
- New `clear_triggered_limits` fixture in `conftest.py` for cleaner test isolation
- Manual ETag manipulation and corruption recovery testing capabilities
- Direct server response validation for invalid ETag scenarios

### Enhanced
- ETag caching functionality now thoroughly validated with real server interactions
- Both sync and async client ETag behavior verification across multiple test scenarios
- INI file lifecycle testing (creation, corruption, recovery, validation)
- Test coverage for edge cases including empty ETags, concurrent access, and multiple refresh cycles

## [0.1.7] - 2025-01-30
### Fixed
- Fixed URL serialization issue in `_make_json_serializable` method for httpx.URL objects
- Resolved 422 validation errors when tracking usage for OpenAI-compatible APIs (DeepSeek, Gemini)
- Fixed usage event delivery failures that were preventing proper cost tracking
- All OpenAI and OpenAI-compatible real endpoint tests now pass successfully

## [0.1.6] - 2025-01-27
### Added
- Comprehensive usage limit functionality with real-time enforcement
- `UsageLimitExceeded` exception for blocking API calls when limits are hit
- End-to-end integration testing across multiple LLM providers
- Enhanced event delivery verification in test suite

### Enhanced
- Real endpoint testing infrastructure for Bedrock and other providers
- Triggered limits persistence and configuration management
- Error handling with detailed limit information and blocking behavior

## [0.1.5] - 2025-07-24
### Changed
- Added /dist/* to gitignore
- Updated `TriggeredLimit` structure. Vendor details now provide
  `config_id_list` and `hostname` values. The old `vendor` string field
  has been removed.

## [0.1.3] - 2025-07-24
### Added
- Initial release of the AICostManager Python SDK
- Cost tracking wrapper for popular LLM SDKs
- Background delivery queue with retry logic
- Configuration via environment variables

## [0.1.0] - 2025-07-24
### Added
- False start