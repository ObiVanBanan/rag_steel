"""Compatibility wrapper for CSV profiling."""

from src.rag_steel import data_builder as _impl

DEFAULT_CONFLICT_LIMIT = _impl.DEFAULT_CONFLICT_LIMIT
DEFAULT_REPORT_PATH = _impl.DEFAULT_REPORT_PATH
DEFAULT_SOURCE_PATH = _impl.DEFAULT_SOURCE_PATH
DataProfile = _impl.DataProfile
REQUIRED_COLUMNS = _impl.REQUIRED_COLUMNS
build_source_documents = _impl.build_source_documents
build_source_documents_from_frame = _impl.build_source_documents_from_frame
build_profile_report = _impl.build_profile_report
main = _impl.main
profile_csv = _impl.profile_csv
save_profile_report = _impl.save_profile_report

__all__ = [
    "DEFAULT_CONFLICT_LIMIT",
    "DEFAULT_REPORT_PATH",
    "DEFAULT_SOURCE_PATH",
    "DataProfile",
    "REQUIRED_COLUMNS",
    "build_source_documents",
    "build_source_documents_from_frame",
    "build_profile_report",
    "main",
    "profile_csv",
    "save_profile_report",
]


if __name__ == "__main__":
    from src.rag_steel.data_builder import main

    raise SystemExit(main())
