from abc import ABC, abstractmethod


class FileCollectorInterface(ABC):
    """
    File collector interface, primarily responsible for collecting file paths from the dataset
    and returning a list containing all file paths.
    """
    @abstractmethod
    def collect(self, **kwargs):
        pass

class FileSplitterInterface(ABC):
    """
    File splitter interface, primarily responsible for splitting the collected list of file paths,
    typically into training, validation, and test sets.
    """
    @abstractmethod
    def split(self, **kwargs):
        pass
