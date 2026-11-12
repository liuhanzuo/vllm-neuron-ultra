"""Exact completed Qwen package text pins for Mini historical compatibility.

This is evidence of ended historical source membership, not model success,
current freshness, native authority, source replacement or directory permission.
No IO is performed except through the caller's original authenticated reader.
"""
from pathlib import Path
import re
from research import minicpm5_formal_literal_codec_v1 as codec

ROOT = Path(__file__).resolve().parents[1]
require = codec.require

QWEN_PACKAGE_TEXT = {'/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/__init__.py': '70d4db750a8213758205101d73186af02a14d9fee883303023ae5efd37208fec',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/core/__init__.py': '34c6f6a40914df002a1136804f2627c4d797033b2dc726fabff8a387bb2a6f11',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/core/scheduler.py': 'bf69b12b9ece4235ebfc6fb6bb45a7b67c17aabc2fa37f993cfb8da937594d50',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/distributed/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/__init__.py': 'f4b98116f32de98fc7ac1103dbb9e8ec10805ea3466d4c8fed2a42e2fb9fe68f',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/lfm3/__init__.py': '57be299c865486907f8ceeec7d2c4cf5d9d47d3ea7876e6677936161228b2d2d',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/lfm3/artifacts.py': 'ad799b2a14f26d9e53aa86d73d12437eebb0f89323b6260ca5e3160547bd6286',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/lfm3/cache_writer.py': 'bd1af5ae871baf7deab294670072d4d014ead3318e93025a17bad9824ca05427',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/lfm3/helpers.py': '89cccbdd092b732a44302ea06921102ea41d961860fe9594f18c103ec250b742',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/lfm3/identity.py': 'a712e40e4e83b1fb3bf7bf2b66896760cd1f8ba8776019105054785f6c442b27',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/lfm3/modeling.py': '3da916f38ecf9202dbc3b898e04351f80f1bccf3763807dcef8985b3872fa70e',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/lfm3/nki_write.py': 'b8ef846eb495c559c1b2094d8bc2deb265e34bbce2ad909828aad0dc57f7a9a5',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/lfm3/nki_write_torch.py': 'b359b5b315827f923899b96cd88ad9d1a868bf6d956ee6541362486ac9ac03b3',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/lfm3/prepare_view.py': '9257a8c6199cf647cad45e6b6f22eb52110d4e60c6d1e213ec1e699d6d2ba15f',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/lfm3/provenance.py': 'c1fd318c96aa365ce29041a805ca03bdf9c65e7dc726a28386ff5865702b5da3',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/lfm3/registration.py': '5a50f16089ea91e61d53eae2641054e67d877e06f857466f89a1b689444a5d1e',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/lfm3/text_core.py': '85ac7401dccef0b286460854093285a289f9abe14d38385b75fb33b60b21aecf',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/nemotron9b/__init__.py': 'e2435f7fae27ed68fda952ebffc854204a320e720eb7eca9bdd623283dcd5959',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/nemotron9b/artifacts.py': '23e9512442c235b971dd04c19c9ad2103d699630f3895ae81136050f2c491187',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/nemotron9b/cache.py': '67943c48d429ed3f9ce9113edaa724b3840ed89857849873f511698726312c11',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/nemotron9b/helpers.py': '3ca8d82079ca339834fcf0689c270d9eb0b359e11764f82cea45e8f96b7f75c3',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/nemotron9b/identity.py': 'b95e4e7535cad63e6df707aaa5422ac4ad26879386f9b2c9d0777667636699e4',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/nemotron9b/modeling.py': 'b86ca3656e048d38958d535db5c01de387b2a6897048d9a3f24c354630288665',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/nemotron9b/nki_read.py': '4054bcc17149a6b81139218a5ad7c2b716b6a1c149e418ddbd139cf801dcd579',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/nemotron9b/nki_read_torch.py': 'a8afef312d6f91d0590c71cc0268383bb873c8272cea515e84e74f5f2721be4e',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/nemotron9b/nki_write.py': 'b8ef846eb495c559c1b2094d8bc2deb265e34bbce2ad909828aad0dc57f7a9a5',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/nemotron9b/nki_write_torch.py': 'b359b5b315827f923899b96cd88ad9d1a868bf6d956ee6541362486ac9ac03b3',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/nemotron9b/prepare_view.py': '8c68ec861b6f733f2baa29d9ea77b5b677242009dc86b8d37caf1efa6b75335a',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/nemotron9b/primitives.py': 'a5022c303779dc2bcd50a4959e4a621fed37d9a2f36856b0f3062d2d3dd25594',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/nemotron9b/provenance.py': '451d168eee16995173d2c105ff31e142dbf290cbcc4208c3789b741705970eff',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/nemotron9b/registration.py': '29b230ee18161e3315446f434e405ab870c8459bfd499e0537db747a8b1b1ea9',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/nemotron9b/serve.py': 'a7cf24b490ab3e2efb55214dfaf409e9086ebee19a2001c4ab8dbb987e3b140e',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/nemotron9b/state.py': '634063828f9799823a78a56ad16a65ac955e9878ee9f18fd1c504ca24368df19',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/nemotron9b/text_core.py': 'd26c17bd52e4a087099d7b78f8a80c8adf04d3ec9bae57b65042c4af925e8e24',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/qwen35/__init__.py': 'a07686e18368bfded1b040270c2e335ece72db3aefa15a79020c2ff067a4c28f',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/qwen35/artifacts.py': 'f32a7bffb8752514aa65ca7e0599b3781557bf7974e7b3bec311df473b656091',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/qwen35/build.py': 'e449f71f5759cc861cb02fd3b40d6d3186f64d8d33cf275da1f901d9641514c8',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/qwen35/helpers.py': 'b77c567096ba8869b04ebe9971d2418939752f506a5fce6ccebbc5ea1d9b787d',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/qwen35/identity.py': '4bdf0769b1cfb577480da9bd416310baec254e6300cf41597437c488401a98b9',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/qwen35/modeling.py': '2cffd29a292a5b5115e8c4499f0ab1a24b3388abbf6d9787580138ff918234ad',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/qwen35/prepare_view.py': 'c326d48eee241dc60a1f7a7b779f8bd75182809fb7b1991f7a7d177ad91f566b',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/qwen35/primitives.py': '64db0f2a04dcf41f3747694419d576fa4fb865038fc273c5030bb7604ade81fa',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/qwen35/provenance.py': '682313f7b2e71b6b5bbeb3a11e82190bfa3c77ddd9a209bec36ae85835e77e89',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/qwen35/registration.py': 'bb5a3a97efa7ea253549697c0386c7a3e98d26a76dcf9f47faa79ba259a43ff7',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/qwen35/serve.py': '7a16b34b805075b5eef9771057cbcc3ebb0fb207814072d782fd3aa9212a9265',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/qwen35/state.py': '095e46f6d02dfc4b43cd77ebaa20a7a25a4879d2786b2c1a2ed5a6e73ffbabc7',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/qwen35/text_core.py': 'c9ee19b8e95eae12193e292286804e660424ca5fd0ace1b64c57f4e9d37011ba',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/qwen35/validation.py': '294c81d07aee0396c960afc1d6f78a0da98d838ab05cd970cbe55a8536ab6dff',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/smollm3/__init__.py': '4555128e22c214030432dbdab4e983ec50e7ee84bf818977795a76098bcad00d',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/smollm3/artifacts.py': '4bc4892f971fa9fabed049e13aff0e34a95056e6dcbb8cc4bb3107189670d2f5',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/smollm3/cache_attention.py': '9a92dce147ad8f740864b31a48323d0cc0a37860a24b5e263616eb76a28c33d7',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/smollm3/helpers.py': 'b904d2b8ec9e49a57237114c756166806c534bb614d136bbee48ea6c05a008f3',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/smollm3/identity.py': '31cc6a3993ec0bf1be61cb07f431ed682ecb4d4c678a03d1d3e9af30067645e2',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/smollm3/modeling.py': '8501ad8016a3a7f6396825e9410ed7878083791ef4e0b4542081e9d08243a6a5',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/smollm3/nki_read.py': '4054bcc17149a6b81139218a5ad7c2b716b6a1c149e418ddbd139cf801dcd579',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/smollm3/nki_read_torch.py': 'a8afef312d6f91d0590c71cc0268383bb873c8272cea515e84e74f5f2721be4e',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/smollm3/nki_write.py': 'b8ef846eb495c559c1b2094d8bc2deb265e34bbce2ad909828aad0dc57f7a9a5',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/smollm3/nki_write_torch.py': 'b359b5b315827f923899b96cd88ad9d1a868bf6d956ee6541362486ac9ac03b3',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/smollm3/prepare_view.py': '1c0ac2ed24baa8b43338b271cbaf53cc077855581e7202d8f8f975fd5e7ad3b8',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/smollm3/provenance.py': '62202d9338e7e43262bd73157988041b5db19d04d38d9e6f8fbc05a921c1299b',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/smollm3/registration.py': '438a3b4abe2f9eb70b5a56fc50ce65f0c25df8c0fd25137d5d99e32ea26224ec',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/smollm3/text_core.py': '97fd88e4ca0ffbd470328d5359216ad4ba364f3b2e949a5c0e6775bd7e583527',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/platform.py': '4387dfe509f055d72b93d955c1fb103142e000c452898680bb71e3e0886f8f26',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/platform_overrides.py': '01e81c6ec753ced45dd57bded421d0909ec930b0cd61ec25382b7dfb711aba6d',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/worker/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/worker/constants.py': '8342a9cfcd077b3c3b5d8ad23254fa3080a1f30c02331a7889c21965ad9726a3',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/worker/cpu_sampler.py': '9638cd6189a819684efe02fb25b959065c9c5ea5cfc3c45ea3a9153b9ce4289a',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/worker/neuron_worker.py': '3f3c2195e1fc399e56b701d63a6d1f67f46d6bcaef38b42ffce095fe1a1201ba',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/worker/neuronx_distributed_model_loader.py': 'c1881ab64a296bb6a98657b8a8135ee8facc954c5e684803f2624dd2e2dcbb31',
 '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/worker/neuronx_distributed_model_runner.py': '4484edd883d199310e1646969417e05a8162148e404f840dc66cad6c6233a77e'}

HISTORICAL_CATALOG = {'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v98.json',
 'sha256': 'e04e68b609fac07d600ed6fbb34a62e0c08b85e9b98207a666f916e7096fece6'}

ENDED = [{'bundle': {'bytes': 19612,
             'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-package-native-preflight-source-bundle-20261112-v3.json',
             'sha256': '6c45ed5293eca2cd7ee48488ebd64f79e26c12136b2aec8b46af40c28238a080'},
  'index': 145,
  'review': {'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-package-preflight-r1b-20261112-review-v2.json',
             'sha256': 'bf57a43fee68f9a833d30e9e477c308ac33a461ede5679f278bd56ec2af9cfd9'},
  'spec': {'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-package-preflight-r1b-20261112-spec-v2.json',
           'sha256': 'fe435bd6fb622cbc84026369f815eaf4cafa1a99ded552fc18f3f3ecd2a0f923'},
  'terminal': {'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-package-preflight-r1-root-actual-exit-20261112-v1.json',
               'sha256': 'b41fcc6ca9f618e85b88302945df7d9c034fdbf1124a2917a1980476389530df'},
  'tool': {'exit_code': 1, 'kind': 'tool_session', 'session_id': 32453, 'terminal_chunk': 'ce28a2'}},
 {'bundle': {'bytes': 19854,
             'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-package-native-preflight-source-bundle-20261112-v4.json',
             'sha256': '217320a7ce35d119601c95a6b38299de909bce64f4fb9543331a52968ae9b82e'},
  'index': 146,
  'review': {'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-package-preflight-r2-20261112-review-v3.json',
             'sha256': '37179a07ded1737d576d87b2411f120aaefe20b5facfa7bd5cd4ebaa3bbd04d9'},
  'spec': {'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-package-preflight-r2-20261112-spec-v3.json',
           'sha256': '493d29d50d3c4f82e93d86146ef61fa8dfebce5eca088956c15f1eed30ac9988'},
  'terminal': {'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-package-preflight-r2-root-actual-exit-20261112-v1.json',
               'sha256': 'b9aec78adf727cf1869449a3cf1ba9cfea61305f5473c06e6852f5750b55811b'},
  'tool': {'exit_code': 1, 'kind': 'tool_session', 'session_id': 35784, 'terminal_chunk': '60e353'}}]


def authenticate_completed_catalog(catalog_ref, read, guard=lambda: None):
    """Bind ended rows, exact approved spec/review refs and whole small bundles.

    Source freezing checks complete spec/review bytes and their bundle edges.
    Runtime revalidates the pinned real terminal carrying those exact refs and
    the complete fixed bundle; it does not reopen the 9.25MB specs or increase
    the original ordinary JSON 8MiB ceiling.
    """
    codec.reference(catalog_ref)
    guard()
    historical = read(HISTORICAL_CATALOG)
    current = historical if catalog_ref == HISTORICAL_CATALOG else read(catalog_ref)
    require(len(historical['runs']) == 147 and type(current['runs']) is list
            and len(current['runs']) >= 147 and current['runs'][:147] == historical['runs'],
            'Every exact original v98 row retained; no partial or rewritten historical catalog')
    for proof in ENDED:
        row = historical['runs'][proof['index']]
        require(row['terminal'] == proof['terminal'] and row['tool'] == proof['tool'],
                'Both exact externally ended failed Qwen tool rows')
        ended = read(proof['terminal'])
        require(ended['schema'] == 'qwen35_package_preflight_root_actual_exit_v1'
                and ended['reviewer'] == '/root' and ended['actual_tool'] == row['tool']
                and ended['spec'] == proof['spec'] and ended['review'] == proof['review']
                and ended['owners'] == row['owners'] and ended['units'] == row['units']
                and ended['actual_tool_exit_observed'] is True and ended['actual_outer_exit_observed'] is True
                and ended['physically_drained'] is True and ended['owners_gone'] is True
                and ended['original_lock_reacquired_and_released'] is True
                and ended['lock_identity'] == {'device': 66306, 'inode': 8294421}
                and ended['native_preflight_passed'] is False
                and ended['full_model_qualified'] is False and ended['new_speedup_claim'] is False,
                'Actual failed/drained historical terminal, never promoted to model/native success')
        bundle = read(proof['bundle'])
        require(bundle['checkout'] == '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107'
                and bundle['source_commit'] == 'e40fd753ba080a9a4640688f4d6beda9985b7099'
                and {bundle['checkout'] + '/' + path: sha for path, sha in bundle['package_sources'].items()} == QWEN_PACKAGE_TEXT,
                'Exact seventy package text pairs from each original bundle')
        require(all(row['source_sha256'].get(path) == sha for path, sha in QWEN_PACKAGE_TEXT.items()),
                'All seventy exact pins present in each independently ended row')
        guard()
    return dict(QWEN_PACKAGE_TEXT)


def extension_shape(extension, *, completed_lfm_text, catalog_ref, read, guard=lambda: None):
    """Original bounded text domain plus this one proved seventy-pin history."""
    require(type(extension) is dict and 0 < len(extension) <= 2048, 'Original bounded additive extension')
    completed = {}
    for path, sha in extension.items():
        require(type(path) is str and Path(path).is_absolute() and str(Path(path)) == path
                and '..' not in Path(path).parts and Path(path).suffix in ('.py', '.json', '.md', '.txt', '.sh')
                and type(sha) is str and re.fullmatch('[a-f0-9]{64}', sha), 'Original exact canonical additive text pin')
        if path.startswith(str(ROOT) + '/') or completed_lfm_text.get(path) == sha:
            continue
        require(QWEN_PACKAGE_TEXT.get(path) == sha, 'Only exact ended Qwen package path and SHA; no new external permission')
        completed[path] = sha
    if completed:
        require(completed == QWEN_PACKAGE_TEXT, 'Complete seventy-pin ended history, never a selected subset')
        require(authenticate_completed_catalog(catalog_ref, read, guard) == completed,
                'Shared preparation and actual runtime evidence binding')
    return extension
