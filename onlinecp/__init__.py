try:
    import pkg_resources
    package = pkg_resources.get_distribution('lightonml')
    __version__ = package.version
except pkg_resources.ResolutionError:
    __version__ = ""
