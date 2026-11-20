import csv
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np


def normalise(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or not np.isfinite(vector).all() or norm < 1e-8:
        raise ValueError("Invalid embedding")
    return vector / norm


class Gallery:
    def __init__(self, path, model_sha256, embedding_dim):
        self.path = Path(path)
        self.model_sha256 = model_sha256
        self.embedding_dim = embedding_dim
        self.people = {}
        if self.path.is_file():
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if value["model_sha256"] != model_sha256:
                raise ValueError("Gallery uses a different face checkpoint. Re-register with the selected model.")
            for person_id, person in value["people"].items():
                vector = normalise(person["embedding"])
                if len(vector) != embedding_dim:
                    raise ValueError("Gallery embedding dimension differs from model")
                self.people[person_id] = {**person, "embedding": vector.tolist()}

    def register(self, person_id, name, embeddings):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", person_id):
            raise ValueError("ID must contain 1-48 letters, numbers, underscores, or hyphens")
        if not name.strip():
            raise ValueError("Name cannot be empty")
        vectors = [normalise(vector) for vector in embeddings]
        if not vectors or any(len(vector) != self.embedding_dim for vector in vectors):
            raise ValueError("Invalid registration embeddings")
        template = normalise(np.mean(vectors, axis=0))
        self.people[person_id] = {"name": name.strip(), "embedding": template.tolist(),
                                  "samples": len(vectors), "registered_at": datetime.now().astimezone().isoformat()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"model_sha256": self.model_sha256, "people": self.people},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def match(self, embedding, metric, threshold):
        if not self.people:
            return None, "Unknown", None
        vector = normalise(embedding)
        identities = list(self.people)
        templates = np.array([self.people[key]["embedding"] for key in identities], dtype=np.float32)
        if metric == "cosine":
            scores = templates @ vector
        elif metric == "negative_euclidean":
            scores = -np.linalg.norm(templates - vector, axis=1)
        else:
            raise ValueError("Unknown similarity metric")
        index = int(np.argmax(scores))
        score = float(scores[index])
        person_id = identities[index]
        if score < threshold:
            return None, "Unknown", score
        return person_id, self.people[person_id]["name"], score


class AttendanceLog:
    fields = ["timestamp", "person_id", "name", "score", "liveness", "mode", "model_sha256"]

    def __init__(self, path, model_sha256):
        self.path = Path(path)
        self.model_sha256 = model_sha256
        self.seen = set()
        if self.path.is_file():
            with self.path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    self.seen.add((row["person_id"], row["timestamp"][:10]))

    def record(self, person_id, name, score, liveness, mode="normal", now=None):
        now = now or datetime.now().astimezone()
        key = (person_id, now.date().isoformat())
        if key in self.seen:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and self.path.stat().st_size > 0
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields)
            if not exists:
                writer.writeheader()
            writer.writerow({"timestamp": now.isoformat(), "person_id": person_id,
                             "name": name, "score": f"{score:.5f}", "liveness": liveness,
                             "mode": mode, "model_sha256": self.model_sha256})
        self.seen.add(key)
        return True
