# OpenThoughts Agent Dataset Statistics

## Summary

| Dataset | Size | Description |
|---------|------|-------------|
| **SFT** | 15,209 traces | Conversation traces for supervised fine-tuning |
| **RL** | 728 tasks | Dockerized RL environments with tests |

---

## 🧠 SFT Dataset Details

### Task Type Distribution
| Task Type | Count | Percentage |
|-----------|-------|------------|
| inferredbugs | 8,172 | 53.7% |
| nl2bash | 7,037 | 46.3% |

### Conversation Statistics
| Metric | Value |
|--------|-------|
| Total traces | 15,209 |
| Unique tasks | 15,209 (no duplicates) |
| Mean messages per conversation | 15.2 |
| Median messages | 14 |
| Max messages | 64 |
| Min messages | 2 |

### Token Statistics (Estimated)
| Metric | Value |
|--------|-------|
| **Total tokens** | **106.5M** |
| Assistant tokens | 35.1M |
| Mean tokens per trace | 7,000 |
| Median tokens per trace | 5,903 |

### Key Insights
- **Even split between task types**: 46% nl2bash (command generation) vs 54% inferredbugs (bug fixing)
- **Rich conversations**: Average of 15 messages per trace with some reaching 64 turns
- **inferredbugs are longer**: Mean of 18 messages vs 12 for nl2bash (50% more turns)
- **~106M total tokens**: Substantial dataset for agent fine-tuning

---

## 🎮 RL Dataset Details

### Task Composition (100% coverage)
| Component | Count | Notes |
|-----------|-------|-------|
| Dockerfile | 728 | Every task has a container definition |
| instruction.md | 728 | Task description for the agent |
| solution/ | 729 | Ground truth solutions |
| test.sh | 1,661 | Multiple test files per task |

### Archive Statistics
| Metric | Value |
|--------|-------|
| Total archive size | 9.9 MB |
| Mean task size | 14.2 KB |
| Median task size | 2.3 KB |
| Max task size | 7.2 MB (outlier with many assets) |

### Files per Task
| Metric | Value |
|--------|-------|
| Mean files | 10.0 |
| Min files | 8 |
| Max files | 186 |

### File Type Distribution (Top 10)
| Extension | Count | Description |
|-----------|-------|-------------|
| .txt | 1,524 | Text files, logs |
| .sh | 1,500 | Shell scripts (setup, tests) |
| .md | 780 | Documentation, instructions |
| .json | 728 | Configuration files |
| .toml | 728 | Config (pyproject.toml) |
| Dockerfile | 728 | Container definitions |
| .log | 142 | Log files |
| .gz | 55 | Compressed files |
| .bin | 43 | Binary test data |
| .c | 39 | C source files |

### Key Insights
- **Fully containerized**: Every task has a Dockerfile for reproducible environments
- **Test-driven**: Average 2.3 test files per task for robust verification
- **Diverse file types**: Tasks include C, Java, PHP, Python, and even media files (jpg, mp4)
- **Compact**: 728 complete RL environments in just 9.9 MB

---

## Notable Statistics for Sharing

### TL;DR
```
📊 OpenThoughts Agent Dataset v1

SFT Dataset:
  • 15,209 conversation traces
  • 106.5M total tokens
  • 46% nl2bash + 54% inferredbugs
  • Avg 15 turns per conversation

RL Dataset:
  • 728 dockerized tasks
  • 100% have Dockerfile + instruction + tests + solution
  • 10 files per task on average
  • 9.9 MB total (very compact!)
```

### Interesting Findings

1. **Bug fixing requires more conversation turns**: inferredbugs tasks average 18 messages vs 12 for nl2bash (50% longer)

2. **No duplicate tasks**: All 15,209 SFT traces have unique task IDs

3. **Dense RL format**: 728 complete executable environments in under 10 MB

4. **Full verification coverage**: Every RL task has:
   - A Dockerfile (reproducible environment)
   - instruction.md (agent prompt)
   - test.sh (automated verification)
   - solution/ (ground truth)

5. **Long-tail token distribution**: Some traces reach ~80K tokens, but median is ~6K
