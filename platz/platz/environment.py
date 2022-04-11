"""Classes for representing different environment types"""


from abc import ABCMeta, abstractmethod


class _Environment(ABCMeta):

    pass


class _Environment1D(_Environment):

    pass


class _Environment2D(_Environment):

    pass


class LinearTrack(_Environment1D):

    pass


class CircularTrack(_Environment1D):

    pass


class OpenField(_Environment2D):

    pass
