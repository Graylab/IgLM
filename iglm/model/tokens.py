BAD_WORD_IDS = [[0], [1], [25], [26], [27], [28], [29], [30], [31], [32]]
MASK_TOKEN = '[MASK]'
SEP_TOKEN = '[SEP]'
ANS_TOKEN = '[ANS]'

SPECIES_TO_TOKEN = {
    "Camel": "[CAMEL]",
    "human": "[HUMAN]",
    "HIS-mouse": "[MOUSE]",
    "mouse_Balb/c": "[MOUSE]",
    "mouse_BALB/c": "[MOUSE]",
    "mouse_C57BL/6": "[MOUSE]",
    "mouse_C57BL/6J": "[MOUSE]",
    "mouse_Ighe/e": "[MOUSE]",
    "mouse_Ighg/g": "[MOUSE]",
    "mouse_Igh/wt": "[MOUSE]",
    "mouse_outbred": "[MOUSE]",
    "mouse_outbred/C57BL/6": "[MOUSE]",
    "mouse_RAG2-GFP/129Sve": "[MOUSE]",
    "mouse_Swiss-Webster": "[MOUSE]",
    "rabbit": "[RABBIT]",
    "rat": "[RAT]",
    "rat_SD": "[RAT]",
    "rhesus": "[RHESUS]",
}

CHAIN_TO_TOKEN = {"Heavy": "[HEAVY]", "Light": "[LIGHT]"}
