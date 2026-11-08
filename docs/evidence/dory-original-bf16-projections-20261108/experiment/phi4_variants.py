"""Pinned Phi-4-mini experiment identities, shared by both runtime environments."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VARIANTS = {
    'instruct': 'cfbefacb99257ffa30c83adab238a50856ac3083',
    'reasoning': '0e3b1e2d02ee478a3743abe3f629e9c0cb722e0a',
}
HF_SOURCE_SHA = 'a5736ee53de6f98fc0ae31f3a42085044c946a1c992326bb491c1ef8816e47c8'


def identity(variant):
    revision = VARIANTS[variant]
    source = Path('/mnt/nvme/models') / ('Phi-4-mini-' + variant)
    return {'variant': variant, 'revision': revision, 'source': source,
            'repo': 'microsoft/' + source.name,
            'view': source.with_name(source.name + '-neuron-text-exp20261031'),
            'dest': ROOT / ('phi4-' + variant),
            'case_path': ROOT / ('phi4-mini-' + variant + '-baseline.json')}
