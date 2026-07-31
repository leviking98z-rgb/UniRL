"""Compatibility aliases for the remote VideoAlign reward owner.

Heavy, non-differentiable VideoAlign inference lives in
``unirl-reward-service``. New recipes should target
``unirl.reward.remote.RemoteRewardBackend`` and ``RemoteRewardSpec``
directly. These aliases keep the historical import paths resolvable for one
deprecation cycle without carrying a second model implementation in core.
"""

from unirl.reward.remote import RemoteRewardBackend, RemoteRewardSpec

VideoAlignRewardScorer = RemoteRewardBackend
VideoAlignSpec = RemoteRewardSpec

__all__ = ["VideoAlignRewardScorer", "VideoAlignSpec"]
