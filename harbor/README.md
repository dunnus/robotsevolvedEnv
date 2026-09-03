# Harbor task collection

Every task is a self-contained Harbor schema 1.4 environment with `network_mode = "no-network"`, a local puzzle, an exploration CLI, an oracle solution, and a reward-producing verifier.

## Lesson tasks

The public Lessons mode contains 12 records. Records 4 and 7 are duplicate copies of Moving used by the website as section separators, leaving these 10 playable boards:

| Section | Lesson | Task directory | Move limit |
| --- | --- | --- | ---: |
| Beginner | Moving | `robotsevolved-lesson-beginner-moving` | 20 |
| Beginner | Stacking | `robotsevolved-lesson-beginner-stacking` | 20 |
| Beginner | Stacking 2 | `robotsevolved-lesson-beginner-stacking-2` | 20 |
| Beginner | Stopping on Goal | `robotsevolved-lesson-beginner-stopping-on-goal` | 20 |
| Intermediate | Stacking | `robotsevolved-lesson-intermediate-stacking` | 20 |
| Intermediate | Infinite Stacking | `robotsevolved-lesson-intermediate-infinite-stacking` | 46 |
| Advanced | Total Bot Requirement | `robotsevolved-lesson-advanced-total-bot-requirement` | 43 |
| Advanced | Shifting Bots | `robotsevolved-lesson-advanced-shifting-bots` | 20 |
| Advanced | Shuffling Bots | `robotsevolved-lesson-advanced-shuffling-bots` | 20 |
| Advanced | Group Movement | `robotsevolved-lesson-advanced-group-movement` | 70 |

Run any directory with the Harbor Oracle first, for example:

`harbor run -p ./harbor/robotsevolved-lesson-beginner-moving -a oracle`

The website was used only to curate the static board snapshot in `data/lessons/lessons.json`. Containers and agents never access the site.