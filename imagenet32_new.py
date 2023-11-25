import tensorflow as tf
import tensorflow_datasets as tfds
import pickle
import numpy as np

"""Downsampled Imagenet dataset. (new version, adapted by Kaiwen Zheng)"""

_CITATION = """\
@article{DBLP:journals/corr/OordKK16,
  author    = {A{\"{a}}ron van den Oord and
               Nal Kalchbrenner and
               Koray Kavukcuoglu},
  title     = {Pixel Recurrent Neural Networks},
  journal   = {CoRR},
  volume    = {abs/1601.06759},
  year      = {2016},
  url       = {http://arxiv.org/abs/1601.06759},
  archivePrefix = {arXiv},
  eprint    = {1601.06759},
  timestamp = {Mon, 13 Aug 2018 16:46:29 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/OordKK16},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """\
Dataset with images of 2 resolutions (see config name for information on the resolution).
It is used for density estimation and generative modeling experiments.
For resized ImageNet for supervised learning ([link](https://patrykchrabaszcz.github.io/Imagenet32/)) see `imagenet_resized`.
"""

_DL_URL = "http://image-net.org/small/"

_DATA_OPTIONS = ["32x32"]


class DownsampledImagenetConfigNew(tfds.core.BuilderConfig):
    """BuilderConfig for Downsampled Imagenet."""

    def __init__(self, *, data=None, **kwargs):
        """Constructs a DownsampledImagenetConfig.
        Args:
          data: `str`, one of `_DATA_OPTIONS`.
          **kwargs: keyword arguments forwarded to super.
        """
        if data not in _DATA_OPTIONS:
            raise ValueError("data must be one of %s" % _DATA_OPTIONS)

        super(DownsampledImagenetConfigNew, self).__init__(**kwargs)
        self.data = data


class DownsampledImagenetNew(tfds.core.GeneratorBasedBuilder):
    """Downsampled Imagenet dataset."""

    BUILDER_CONFIGS = [
        DownsampledImagenetConfigNew(  # pylint: disable=g-complex-comprehension
            name=config_name,
            description=("A dataset consisting of Train and Validation images of " + config_name + " resolution."),
            version=tfds.core.Version("2.0.0"),
            data=config_name,
            release_notes={
                "2.0.0": "New split API (https://tensorflow.org/datasets/splits)",
            },
        )
        for config_name in _DATA_OPTIONS
    ]

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(encoding_format="png"),
                }
            ),
            supervised_keys=None,
            homepage="http://image-net.org/small/download.php",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        pass

    def _generate_examples(self, archive):
        pass
