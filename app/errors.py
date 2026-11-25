"""Domain errors. The API layer maps each one to an HTTP status code."""


class DocQAError(Exception):
    """Base class for all expected, user-facing errors."""


class UnsupportedFileType(DocQAError):
    """The uploaded file extension is not one we can parse."""


class DocumentParseError(DocQAError):
    """The file looked supported but could not be read (corrupt, encrypted, ...)."""


class EmptyDocument(DocQAError):
    """The file parsed fine but had no extractable text (e.g. a scanned PDF)."""


class FileTooLarge(DocQAError):
    """The upload is bigger than MAX_UPLOAD_MB."""


class DocumentNotFound(DocQAError):
    """No stored document has the requested id."""


class LLMError(DocQAError):
    """The language-model provider failed or is misconfigured."""
