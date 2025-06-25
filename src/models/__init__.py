from .glm import GLMAssistant
from .qwen_omni import QwenOmniAssistant
from .minicpm import MiniCPMAssistant
from .gpt4o import GPT4oAssistant
from .naive_qwen import NaiveQwenAssistant

model_cls_mapping = {
    'glm': GLMAssistant,
    'qwen2p5-omni': QwenOmniAssistant,
    'minicpm': MiniCPMAssistant,
    'gpt4o': GPT4oAssistant,
    'naive-qwen': NaiveQwenAssistant,
}
