import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Post templates by mood level (0=calm, 1=moderate distress, 2=high distress)
_TITLES = {
    0: [
        "Having a decent day today",
        "Small wins matter",
        "Tried something new today",
        "Feeling okay for once",
        "Made it through the week",
        "A good therapy session today",
        "Starting to see some improvement",
        "Grateful for the little things",
        "Finally slept well last night",
        "Went outside today",
    ],
    1: [
        "Struggling lately but trying",
        "Does anyone else feel this way",
        "Bad day at work, feeling low",
        "Can't seem to shake this feeling",
        "Need advice on coping",
        "Feeling anxious about everything",
        "Having trouble sleeping again",
        "Therapy isn't helping much",
        "Feeling disconnected from friends",
        "Another rough day",
    ],
    2: [
        "I can't do this anymore",
        "Everything feels hopeless",
        "No point in trying",
        "I feel like giving up",
        "Trapped and can't see a way out",
        "Nobody understands what I'm going through",
        "I'm a burden to everyone",
        "Can't stop crying",
        "The darkness won't go away",
        "I don't want to be here anymore",
    ],
}

_BODIES = {
    0: [
        "I've been working on myself and today I actually felt a bit better. Nothing huge, just a moment where I didn't feel the weight on my chest. Has anyone else experienced these little moments of clarity?",
        "Went for a walk today and it helped. I know it sounds simple but getting out of bed was hard enough. Just wanted to share that small steps count.",
        "My therapist suggested journaling and I've been doing it for two weeks now. I think it's actually helping me process things better. Anyone else journal?",
        "Had a conversation with a friend today about how I've been feeling. They were really supportive. It reminded me that people do care, even when my brain tells me otherwise.",
        "Managed to cook a proper meal today instead of just eating cereal. It's the small victories, right?",
    ],
    1: [
        "I've been feeling really down lately. Work has been stressful and I can't seem to find any motivation. I used to enjoy things but now everything feels like a chore. Does anyone have tips for dealing with this?",
        "My anxiety has been through the roof this week. I keep worrying about things that haven't even happened. I know it's irrational but I can't stop the thoughts. Anyone else deal with this?",
        "I haven't been sleeping well and it's making everything worse. I lie in bed for hours with my mind racing. I've tried meditation but it doesn't seem to work for me. What helps you all?",
        "Feeling really isolated lately. I moved to a new city and don't have many friends here. The loneliness is getting to me. I try to reach out but social anxiety makes it so hard.",
        "Had a panic attack at the grocery store today. I just froze and couldn't move. I'm so tired of anxiety controlling my life. Does anyone have advice for dealing with panic attacks in public?",
    ],
    2: [
        "I don't know how much longer I can keep going. Every day feels the same. Wake up, feel terrible, pretend I'm fine, go to sleep, repeat. I'm exhausted from pretending. I feel completely hopeless about my future.",
        "I feel like I'm drowning and nobody can see it. I put on a brave face but inside I'm falling apart. I don't see things getting better. I've tried therapy, medication, everything. Nothing helps. I'm just tired of fighting.",
        "I can't stop the negative thoughts. They tell me I'm worthless, that nobody would miss me, that I'm a burden. I know rationally that might not be true but it feels so real. I'm scared of my own mind.",
        "Everything in my life is falling apart. Lost my job, my relationship ended, and I can barely afford rent. I feel like I'm trapped in a nightmare with no way out. The hopelessness is overwhelming. I don't know what to do anymore.",
        "I've been crying for hours and I can't stop. The emptiness inside me is consuming everything. I used to have dreams and goals but now I can't even imagine a future. I feel dead inside. I'm reaching out because I don't know where else to turn.",
    ],
}


def generate_synthetic_data(
    config: dict,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    random.seed(seed)

    n_weeks = config.get("synthetic", {}).get("n_weeks", 104)
    posts_range = config.get("synthetic", {}).get("posts_per_week_range", [80, 200])
    crisis_freq = config.get("synthetic", {}).get("crisis_frequency", 0.12)
    subreddits = config["reddit"]["subreddits"]

    start_date = datetime(2024, 1, 1)
    results = {}

    for subreddit in subreddits:
        posts = []
        author_pool = [f"user_{i}" for i in range(500)]

        # Pre-generate crisis weeks
        crisis_weeks = set()
        for w in range(n_weeks):
            if rng.random() < crisis_freq:
                crisis_weeks.add(w)

        for week_idx in range(n_weeks):
            week_start = start_date + timedelta(weeks=week_idx)
            n_posts = rng.randint(posts_range[0], posts_range[1])

            # Determine mood distribution for this week
            is_precursor = (week_idx + 1) in crisis_weeks
            is_crisis = week_idx in crisis_weeks

            if is_crisis:
                mood_probs = [0.15, 0.35, 0.50]  # heavy distress
            elif is_precursor:
                mood_probs = [0.25, 0.40, 0.35]  # building distress
            else:
                mood_probs = [0.50, 0.35, 0.15]  # mostly calm

            for j in range(n_posts):
                mood = rng.choice([0, 1, 2], p=mood_probs)
                title = random.choice(_TITLES[mood])
                body = random.choice(_BODIES[mood])

                # Add variation to post length
                if rng.random() < 0.3:
                    body = body + " " + random.choice(_BODIES[mood])

                post_time = week_start + timedelta(
                    days=rng.randint(0, 7),
                    hours=rng.randint(0, 24),
                    minutes=rng.randint(0, 60),
                )

                # More late-night posts during crisis weeks
                if is_crisis and rng.random() < 0.4:
                    post_time = post_time.replace(hour=rng.randint(22, 24) % 24)

                posts.append({
                    "post_id": str(uuid.uuid4())[:12],
                    "created_utc": int(post_time.timestamp()),
                    "title": title,
                    "selftext": body,
                    "score": max(1, int(rng.normal(10, 8))),
                    "num_comments": max(0, int(rng.normal(5, 4))),
                    "subreddit": subreddit,
                    "author": random.choice(author_pool),
                    "is_self": True,
                })

        df = pd.DataFrame(posts)
        df["created_utc_dt"] = pd.to_datetime(df["created_utc"], unit="s")
        results[subreddit] = df

    return results
