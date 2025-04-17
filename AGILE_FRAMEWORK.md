# Agile Collaboration Framework

## Introduction

This document outlines a flexible, agile framework for collaborating on projects. It's designed for **logical sprints** (focused work blocks) rather than strict time limits, making it suitable for:

- **Learning Projects:** Acquiring new skills or knowledge.
- **Research Efforts:** Investigating topics or questions.
- **Task Breakdown:** Managing complex tasks or projects.

**Goal:** To provide a clear structure for defining goals (Epics), breaking them down into manageable steps (Sprints), executing the work, and tracking progress and understanding.

## 📁 Project Structure

A recommended file structure to organize the project:

```
project_root/
├── README.md                 # Project overview, setup, main goals
├── AGILE_FRAMEWORK.md        # This document
├── data/                     # Directory for datasets and experiments
├── models/                   # Directory for saved model weights and checkpoints
├── sprints/                  # Directory for all work sprints
│   ├── 01_sprint_name/       # Example completed sprint
│   │   ├── README.md         # Goals, tasks, findings
│   │   ├── docs/             # Working notes, logs, documentation
|   |   |   ├── 01_docname.md # etc            
│   │   └── code/             # Outputs, code, practice files
|   |   |   ├── filename.py   # no numbers
└── backlog.md                # Ideas for future sprints
└── progress.md               # Log of our progress and sprint notes
```

_(Adapt this structure as needed)_

## Epic: [Overall Goal / Research Question / Project Title]

**Vision Statement:**
[Clearly state the ultimate objective. What are we trying to achieve, learn, or build?]

**Key Objectives / Success Criteria:**

- [ ] [Measurable outcome or key question 1]
- [ ] [Measurable outcome or key question 2]
- [ ] [Measurable outcome or key question 3]

## Sprints:

**Sprint Goal:**
[What is the primary objective for this specific sprint? What key step towards the Epic are we taking?]

**Tasks / Learning Objectives:**

1.  [ ] [Specific task, concept, or experiment 1]
2.  [ ] [Specific task, concept, or experiment 2]
3.  [ ] [Specific task, concept, or experiment 3]

**Definition of Done / Key Questions Answered:**

- [ ] [Deliverable 1 / Understanding achieved 1]
- [ ] [Deliverable 2 / Understanding achieved 2]
- [ ] [Question answered / Hypothesis tested]

## Backlog

_(Managed in `backlog.md`)_

**Potential Next Sprints:**

- [Idea for next logical step 1]
- [Idea for next logical step 2]
- [Unexplored area or prerequisite]

**Planning Process:**

1.  **Review:** Assess the outcome of the previous sprint. What was learned/achieved? Any roadblocks?
2.  **Prioritize:** Select the most logical next sprint from the backlog based on the Epic.
3.  **Define:** Create the `README.md` for the new sprint, detailing goals and tasks.
4.  **Execute:** Work through the sprint tasks.
5.  **Conclude:** Summarize findings, update progress, and archive the sprint.

## Progress & Assessment

_(Managed in `progress.md` and sprint `README.md` files)_

**Tracking Method:**

- **Sprint READMEs:** Document goals, tasks, findings, and challenges for each sprint.
- **`progress.md`:** Record significant achievements, decisions, or completed phases.

## Adaptability

This framework is designed to be flexible:

- **Sprint Scope:** Adjust the size and complexity of sprints based on progress and understanding.
- **Direction Changes:** If research or learning leads in a new direction, update the Epic and Backlog.
- **Tooling:** Use the file structure and tracking methods that work best for the specific project.

---

_This framework helps structure our collaboration. We will refer to it to plan sprints and track progress._
