"""
persona.py
==========
4차원 Persona schema 정의 (논문 Table: Social Role / Big-Five / Background / Interests).

이 파일은 '정답(ground-truth)'의 구조를 정의합니다. 합성 데이터 생성기(datagen.py)가
이 schema에서 persona를 샘플링하고, BIG-Net(model.py)이 bridging graph로부터 이 4차원을
역으로 예측(추론)합니다.

핵심 설계 의도
--------------
논문 §3.2의 가설을 데이터에 심습니다:
  - 구조적/기술적 role (엔지니어 등) -> Association 관계(part-of, member-of) 밀도 ↑
  - 서사적/공감적 role (교사/작가 등) -> Characterization 관계(cause-of, theme, ...) 밀도 ↑
이 신호가 graph에 존재하므로, GNN이 graph만 보고 persona를 복원할 수 있어야 합니다.
(신호가 약한 Background는 일부러 어렵게 만들어, 논문이 보고한 'background에서 오류가 많다'는
 현상을 재현합니다.)
"""

# ---- 관계 유형 (Irmer 2011의 7가지) --------------------------------------
RELATION_TYPES = [
    "part-of",    # 0  } Association (Indirect Reference by Association)
    "member-of",  # 1  }
    "instrument", # 2  } Characterization (Indirect Reference by Characterization)
    "theme",      # 3  }
    "cause-of",   # 4  }
    "in",         # 5  }
    "temporal",   # 6  }
]
NUM_RELATIONS = len(RELATION_TYPES)
ASSOCIATION_RELS = [0, 1]              # part-of, member-of
CHARACTERIZATION_RELS = [2, 3, 4, 5, 6]

# ---- Dimension 1: Social Role -------------------------------------------
# 각 role에 'mode' 성향(1=구조적/Association, 0=서사적/Characterization)을 부여.
SOCIAL_ROLES = [
    "Software Engineer", "Data Scientist", "Accountant", "Civil Engineer",  # 구조적
    "Teacher", "Novelist", "Therapist", "Journalist",                       # 서사적
]
ROLE_STRUCTURAL = {  # role -> 구조성 점수 [0,1]
    "Software Engineer": 0.95, "Data Scientist": 0.90, "Accountant": 0.85, "Civil Engineer": 0.88,
    "Teacher": 0.35, "Novelist": 0.10, "Therapist": 0.20, "Journalist": 0.30,
}
NUM_ROLES = len(SOCIAL_ROLES)

# ---- Dimension 2: Personality (Big-Five, 연속값 [0,1]) -------------------
BIG5 = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
NUM_BIG5 = len(BIG5)

# ---- Dimension 3: Background (다중 속성, 범주형) -------------------------
EDUCATION = ["HighSchool", "Bachelor", "Master", "PhD"]
LOCATION = ["Urban", "Rural"]
FAMILY = ["Single", "Married", "LivingAlone"]

# ---- Dimension 4: Interests (다중 라벨) ----------------------------------
HOBBIES = ["reading", "traveling", "gaming", "cooking", "sports"]
VALUES = ["creativity", "family", "integrity"]
COMM_STYLE = ["direct", "emotional"]
INTEREST_LABELS = HOBBIES + VALUES + COMM_STYLE   # multi-label 대상
NUM_INTERESTS = len(INTEREST_LABELS)

# ---- 개념(node) 어휘 사전 -----------------------------------------------
# bridging graph의 노드가 될 concept들. 일부는 role/interest와 연결되어 신호를 제공.
ROLE_CONCEPTS = {
    "Software Engineer": ["system", "module", "code", "server", "bug"],
    "Data Scientist":    ["dataset", "model", "feature", "metric", "pipeline"],
    "Accountant":        ["ledger", "audit", "tax", "budget", "invoice"],
    "Civil Engineer":    ["bridge", "beam", "concrete", "blueprint", "load"],
    "Teacher":           ["student", "lesson", "class", "exam", "curriculum"],
    "Novelist":          ["character", "plot", "scene", "chapter", "manuscript"],
    "Therapist":         ["client", "emotion", "session", "trauma", "coping"],
    "Journalist":        ["source", "story", "deadline", "interview", "headline"],
}
INTEREST_CONCEPTS = {
    "reading": ["book", "novel"], "traveling": ["journey", "airport"],
    "gaming": ["game", "quest"], "cooking": ["recipe", "kitchen"],
    "sports": ["match", "training"],
    "creativity": ["idea", "design"], "family": ["home", "parent"],
    "integrity": ["honesty", "rule"],
    "direct": ["fact", "point"], "emotional": ["feeling", "heart"],
}
# 배경/공용 개념(신호 약함) - graph를 자연스럽게 채우는 filler
GENERIC_CONCEPTS = [
    "time", "place", "person", "work", "day", "thing", "way", "world",
    "morning", "evening", "city", "town", "friend", "plan", "goal",
]

# 전체 vocab (concept string -> index). 모델의 node embedding table 크기 결정.
_ALL_CONCEPTS = sorted(set(
    [c for cs in ROLE_CONCEPTS.values() for c in cs] +
    [c for cs in INTEREST_CONCEPTS.values() for c in cs] +
    GENERIC_CONCEPTS
))
CONCEPT2IDX = {c: i for i, c in enumerate(_ALL_CONCEPTS)}
IDX2CONCEPT = {i: c for c, i in CONCEPT2IDX.items()}
VOCAB_SIZE = len(_ALL_CONCEPTS)


def dims_summary():
    return {
        "roles": NUM_ROLES, "big5": NUM_BIG5,
        "education": len(EDUCATION), "location": len(LOCATION), "family": len(FAMILY),
        "interests": NUM_INTERESTS, "relations": NUM_RELATIONS, "vocab": VOCAB_SIZE,
    }


if __name__ == "__main__":
    print("Persona schema dimensions:", dims_summary())
    print("Vocab size:", VOCAB_SIZE)
