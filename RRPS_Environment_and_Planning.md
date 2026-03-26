# Restricted Rock–Paper–Scissors (Kaiji)
*AI Environment Specification & Strategies*
■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ 

## 1. Executive framing
For this project, Restricted Rock-Paper-Scissors should be modeled as a partially observable multi-agent game with social and economic layers, not as a standard single-agent reinforcement-learning benchmark. The canon gives every contestant twelve cards (four Rock, four Paper, four Scissors), three stars, and a four-hour limit. Players simultaneously reveal one card per match, the loser hands over one star, draws return stars to their owners, and used cards are discarded. To clear the game, a contestant must use all cards and finish with at least three stars. Canon also includes war-fund loans, interest pressure, surplus-star cash-out, and a short postgame star market for desperate survivors.[C1][C2][C3][C4]
Because we are comparing personalities rather than training one monolithic agent against random play, the cleanest architecture is: environment + library of strategy families + optimizer over family parameters + population simulator. That lets “survival-first,” “profit-first,” “predatory,” “loyalist,” and other personalities optimize different scalarizations of the same shared outcome vector.
## 2. Canonical rules and environment assumptions
The table below separates source-backed facts from explicit environment conventions. Anything labeled “environment convention” is a formalization chosen for simulation clarity, even when it is strongly suggested by the story.
| Topic | Canon-backed fact | Environment convention |
| --- | --- | --- |
| Initial resources | Each player starts with 12 cards: 4 Rock, 4 Paper, 4 Scissors, plus 3 stars.[C1][C2] | Version 3 also tracks money, debt, and later market access. |
| Round protocol | Players select a card, place it face down, then reveal. Winner takes 1 star; on a draw each keeps their star.[C1] | A played card is consumed in all outcomes and moved to discard/public history. |
| Win condition | Use all cards and finish with 3 or more stars within the time limit.[C1][C2] | “Survive” = terminal state with cards=0 and stars>=3. |
| Loss condition | Fail if cards remain at time-out or if terminal stars are 2 or fewer.[C1] | In simulation, star-bust can trigger immediate failure or forced dependence on a bailout market, depending on version. |
| Money / debt | Contestants borrow war funds; loans compound at 1.5% every 10 minutes across a 4-hour voyage. Money beyond repayment is kept.[C2][C3] | Version 3 lets agents choose initial borrowing, tracks debt continuously/discretely, and scores debt-adjusted wealth. |
| Finishing early | Finishing earlier reduces interest burden.[C1] | Time-to-finish is therefore a meaningful metric, not just cosmetic. |
| Surplus stars | Winners can exchange extra stars for cash or sell them later.[C3] | Finished players may exit to a seller pool or remain active sellers in the market phase. |
| Endgame star market | Players with too few stars get about 10 minutes to buy stars; prices can rise.[C3] | Version 3 models this as an explicit market with bids, asks, bargaining, and emergency rescues. |
| Alliances / coordination | The story explicitly shows teaming, plans built around coordinated draws, card reshuffling, and strategic trades/withholding.[C1][C5] | Versions 2–3 include negotiation, trust, and optional card or star trades as actions. |

## 3. Formal game model recommendation
Model the entire game as a partially observable stochastic game. Every agent has private information (its own hand, trust map, intentions, cash reserves, possible hidden deals) and observes a mixture of public and private signals. The simplest version collapses this into repeated 1v1 simultaneous moves; later versions add social interaction, waiting, observation of the floor-wide card economy, and finally money/debt.
- Private state: own remaining card counts, stars, money/debt, existing agreements, trust beliefs, hidden priorities.
- Public state: time remaining, public card totals, revealed-play history, who is active/finished/eliminated, public market offers.
- Hidden state about others: exact unseen hand composition, real objective weights, betrayal likelihood, off-path plans.
- Observation = private state + public state + any local observations earned by interacting with or watching specific players.
- Transition dynamics are jointly determined by simultaneous match actions, negotiation outcomes, time advancement, and market clearing.
- Terminal utility should usually be computed from a vector of outcomes, then scalarized per personality family rather than forcing a single universal reward.
## 4. Outcome vector and personality-specific scoring
Use a common outcome vector for every run, then let each personality family define its own scalarization over time. This keeps experiments comparable while preserving the fact that different personalities actually care about different things instead of all accidentally optimizing toward the same goal.
| Metric | Description | Why it matters |
| --- | --- | --- |
| Survival | 1 if cards=0 and stars>=3 at legal termination; else 0 | The most canon-faithful primary objective. |
| Final stars | Stars at legal terminal state | Captures buffer, bargaining power, and predatory extraction. |
| Time-to-finish | How early the player exits or locks survival | Important because interest/debt pressure rewards early completion.[C1][C2] |
| Net cash | Cash on hand after star sales and trades | Needed for profit-seeking and brokerage personalities. |
| Debt-adjusted wealth | Cash minus outstanding debt / repayment obligation (note: it may be interesting to give players random amounts of debt coming into the game, affecting their desperation and thus, their tactics) | Essential in Version 3. |
| Market P&L | Profit specifically from buying/selling stars | Measures broker/shark behavior. |
| Trust / reputation | How often agreements were honored and recognized | Important in social versions with repeated interactions. |
| Exploitative yield | Stars/cash gained from players in panic or mathematically dead states | Separates predators from cooperators. |
| Robustness | Average performance across heterogeneous populations | Useful for deciding if a strategy is generally good or only meta-specific. |

A generic scalarization can be written conceptually as:

```text
U = w_survive·Survival + w_stars·FinalStars + w_time·FinishBonus + w_cash·NetCash + w_debt·Debt + ... + w_rep·Reputation + w_pred·ExploitYield
```
Different strategy families simply use different weights, thresholds, or non-linear bonuses.
| Personality family | Example utility emphasis | Behavior encouraged |
| --- | --- | --- |
| Survivor | Huge weight on Survival; small bonus for stars above 3 | Safe draws, trustworthy alliances, avoiding unnecessary risk. |
| Wealth maximizer | Strong weight on debt-adjusted wealth and finish time | Fast completion, surplus-star monetization, efficient trades. |
| Predator | Survival + large star/cash bonuses + exploit yield | Preying on short-star players, forced sells, card cornering. |
| Broker | Survival + market P&L + reputation | Stable dealing, liquidity provision, two-sided trades. |
| Loyal coalitionist | Survival + alliance success + low betrayal penalty | Trust-building and reciprocal play. |
| Chaotic gambler | Variance-seeking utility; tolerance for failure | High-risk matches, deception, volatile endgames. |

## 5. Version A — Simplest 1v1 simulator
Version A is the cleanest place to benchmark micro-strategies. It strips away the crowd, negotiation, and economy, and treats the game as repeated simultaneous-card play between two fixed opponents. This is not the full canon experience, but it is the best sandbox for calibrating card-choice heuristics and parameter tuning.
- Players: exactly 2.
- Episode start: each player gets (R=4, P=4, S=4, Stars=3).
- Turn: both players choose one remaining card simultaneously.
- Resolution: win/loss/draw exactly as canon; both used cards are removed.[C1]
- Episode end: after 12 rounds or earlier if a player becomes mathematically dead under your chosen rule set.
- Suggested success rule: cards=0 and stars>=3.
- Best use: fitting local policies such as balancing, streak-breaking, counter-reading, and intentional draw play.
| Category | Recommended Version A observation fields |
| --- | --- |
| Own hand | r_self, p_self, s_self, stars_self, rounds_left |
| Opponent history | opponent_plays_so_far, recent_window, win/draw/loss streaks |
| Self history | self_plays_so_far, own streaks, card imbalance |
| Optional beliefs | estimated_opponent_type, posterior over next-card distribution |

| Action class | Examples |
| --- | --- |
| Match action | Play Rock / Paper / Scissors from remaining cards |
| Optional meta-action | Early surrender, if your simulator wants mathematically dead states to terminate immediately |
| No social layer | No trading, no waiting, no market, no partner selection |

Suggested Version A baseline policies:
- Uniform random over remaining cards.
- Balance preserver: tries to keep remaining hand counts even.
- Win-stay / lose-shift with bounded repetition.
- Opponent-frequency counter: plays best response to empirical next-card estimate.
- Draw seeker: intentionally preserves mirror possibilities to secure safe exhaustion of cards.
## 6. Version B — Timestepped social-information environment
Version B is where Restricted Rock-Paper-Scissors becomes interesting as a population game. Instead of a fixed opponent, agents live in a crowd with waiting periods between matches, partial observation of the broader floor, and information leakage through repeated play. This is the right version for trust, reputation, selective cooperation, public-card accounting, and reading other players’ trajectories.
- Players: N participants (for example 20, 40, or the story-scale 100+).
- Clock: discrete timesteps; each step may represent a short slice such as 15–30 seconds or 1 minute.
- Cooldown / waiting: after a match or trade, agents may be unavailable for some number of timesteps.
- Public card totals: track remaining Rock/Paper/Scissors still in circulation, if your version exposes that globally.
- Observed play history: agents can watch some or all matches and update beliefs about others’ remaining hands.
- Finished players can move to an inactive/safe state, a seller state, or a spectator/seller hybrid depending on the experiment.
| Observation block | Examples of fields in Version B |
| --- | --- |
| Self | own remaining cards, stars, status, cooldown, alliance memberships |
| Clock | time_step, time_remaining, phase |
| Public economy | public_remaining_R, public_remaining_P, public_remaining_S, active_player_count |
| Visible players | for each seen player: stars, known/estimated remaining cards, last plays, betrayal count, trust score |
| Interaction inbox | incoming match requests, trade offers, alliance proposals, warnings |
| Belief features | scarcity indices, danger score, probability a target is desperate, likelihood of future cooperation |

| Action family | Examples |
| --- | --- |
| Match selection | Request match with player j; accept / refuse request |
| Card play | Conditional on match: choose R / P / S |
| Information actions | Watch a match, share info, withhold info, misreport info |
| Social actions | Propose alliance, honor or betray agreement, blacklist a player |
| Trade actions | Offer card-for-card, card-for-star, or “play-to-draw” coordination plan |
| Time actions | Wait, stall, hunt specific targets, prioritize safe completion |

Version B can still ignore money while being much richer than Version A. The key modeling decision is how much of the floor is observable. Three useful presets are: (i) full public logs, where every revealed card is globally visible; (ii) local observation, where only nearby/watched matches are visible; and (iii) delayed gossip, where information becomes public only if someone shares it.
## 7. Version C — Full economic environment
Version C adds the money system and turns the game into a true social-economy simulation. At this level, agents must care not just about cards and stars, but also debt, interest, liquidity, bailout decisions, and whether to survive at the cost of crushing future repayment.
- Initial borrow decision: choose loan/war-fund size within an allowed range, matching the canon idea of 1–10 million yen borrowed for play.[C2][C3]
- Interest accrual: debt grows at the chosen compounding schedule; a canon-faithful choice is 1.5% every 10 minutes (show rounds to 40% at the end for those who did not end early).[C2][C3]
- Finishing early matters because debt stops growing once the player safely exits, if you choose that convention.
- Postgame market: short-star players can buy stars; extra-star players can sell; prices may spike under desperation.[C3]
- Optional rescue mechanic: buying three stars may free a person from the loser room, matching the story’s aftermath description.[C3]
- Version C is where profit-seeking, predation, brokerage, and debt engineering become first-class behaviors.
| Version C additional state | Examples |
| --- | --- |
| Money | cash_on_hand, reserved_cash, hidden_jewelry/side assets if you want story-style variants |
| Debt | principal, accrued_interest, total_repayment_due, time_since_borrow |
| Market state | current bids/asks for stars, recent sale prices, liquidity depth, desperation index |
| Contract state | promises to repay, private loans, alliance side-payments, future-delivery commitments |
| Terminal wealth | cash + liquidation proceeds - debt - penalties |

A simple canonical debt formula for discrete compounding is:

```text
Repayment(t) = Borrowed × (1.015)^k
```

where `k` is the number of completed 10-minute periods elapsed since borrowing.
## 8. Observables: private vs. public vs. inferred
A major design choice is deciding what counts as an observation versus an inference target. For clean experiments, explicitly separate (a) information given directly to the agent, (b) information that exists publicly but must be aggregated, and (c) information the agent must estimate.
| Type | Examples | Why separate it |
| --- | --- | --- |
| Private observable | Own hand counts, own stars, own money/debt, own agreements | Always known; no inference required. |
| Public observable | Time remaining, revealed cards, active players, public market quotes | Shared information that all agents can exploit. |
| Locally visible | Matches watched, offers received, behavior of nearby targets | Lets information asymmetry emerge naturally. |
| Inferred feature | Estimated remaining hand of target player, betrayal probability, desperation score | This is where personality and modeling style matter most. |
| Latent hidden variable | True internal objective weights of other players | Should not be directly exposed except in oracle baselines. |

Useful derived features for agents in Versions B–C:
- Scarcity index for each card type = (public remaining of type) / (all remaining cards).
- Target desperation score from low stars, high debt, little time remaining, and awkward hand composition.
- Trust score combining honored trades, completed coordinated draws, and betrayal history.
- Exploitability estimate: chance a player must accept a bad offer to survive.
- Endgame parity flags: whether a player can exhaust their hand safely through self-contained draws or must seek outside interaction.
## 9. Strategy families and tunable parameters
Each family can be implemented as a rule-based policy with tunable thresholds, beliefs, and utility weights. Very broad, don’t need to do all.
| Strategy family | Core idea | Typical knobs |
| --- | --- | --- |
| 1. Pure Drawer | Maximize probability of legal completion through repeated draws. | draw_bias, trust_threshold, endgame_panic |
| 2. Safe Buffer Builder | Win just enough early to build a star cushion, then coast to safe exhaustion. | target_star_buffer, aggression_window |
| 3. Balance Reader | Attack opponents who preserve balanced hands and therefore become predictable. | history_window, confidence_cutoff |
| 4. Frequency Counter | Best-respond to empirical next-card frequencies. | prior_strength, smoothing, recency_weight |
| 5. Streak Breaker | Assume opponents avoid repeating patterns; exploit anti-streak psychology. | anti_repeat_weight, deception_tolerance |
| 6. Monotype Hoarder | Accumulate one card type to corner specific matchups or block others’ safe exits. | hoard_target, release_threshold |
| 7. Scarcity Cornerer | Track global card scarcity and stockpile what desperate players need. | scarcity_trigger, max_inventory |
| 8. Coalition Loyalist | Build a small trusted alliance and prioritize joint survival. | coalition_size, betrayal_penalty |
| 9. Selective Loyalist | Cooperate, but only with agents above a trust threshold. | trust_threshold, probation_length |
| 10. Tit-for-Tat | Reciprocate honesty and punish betrayal immediately. | forgiveness_rate, retaliation_horizon |
| 11. Reputation Farmer | Take slightly worse short-run deals to become a preferred counterparty later. | rep_weight, early_generosity |
| 12. Shark Seller | Finish early and extract high prices from desperate short-star players. | min_sale_price, reserve_stars |
| 13. Market Maker | Post both bids and asks to profit from spread, not from all-in predation. | spread, inventory_target |
| 14. Bailout Extractor | Rescue doomed players only at extreme markup or future obligation. | markup, enforceability_weight |
| 15. Debt Minimizer | Rush safe completion because interest is the main enemy. | time_weight, risk_aversion |
| 16. Debt-Fueled Speculator | Borrow aggressively, assuming market or star flipping will cover it. | initial_borrow_frac, leverage_limit |
| 17. Opportunistic Predator | Hunt low-star, awkward-hand, or time-pressured players. | desperation_cutoff, cruelty_weight |
| 18. Feigned Weakness | Pretend to be desperate to secure trust or favorable trades before flipping. | bluff_rate, reveal_policy |
| 19. False-Signal Mixer | Intentionally vary reveals to poison opponent inference. | mixing_noise, pattern_length |
| 20. Watchlist Hunter | Closely track a few players and target them when probabilities become sharp. | watchlist_size, trigger_confidence |
| 21. Endgame Parity Solver | Optimize around odd/even hand exhaustion and last-card constraints. | parity_priority, sacrifice_tolerance |
| 22. Anti-Hoard Enforcer | Punish monopoly behavior by coordinating against cornerers. | collective_action_bias, target_threshold |
| 23. Kingmaker | If unable to maximize own outcome, redirect gains/losses to shape who survives. | spite_weight, ally_bias |
| 24. Vendetta Player | Overweights punishing earlier betrayal even at some cost. | revenge_weight, memory_depth |
| 25. Chaos Gambler | Accepts volatility and low survival for rare big-payoff endings. | variance_preference, panic_threshold |

Good shared parameter knobs across many strategy families:
- risk_aversion: willingness to sacrifice upside for survival probability
- trust_threshold: minimum score needed before accepting a cooperative plan
- betrayal_penalty: how sharply future trust collapses after one betrayal
- scarcity_sensitivity: how aggressively the agent responds to public card shortages
- target_star_buffer: desired stars before switching from aggression to safety
- time_pressure_weight: how much looming interest / time-out changes behavior
- sale_greed: minimum acceptable price to sell surplus stars
- bailout_generosity: willingness to rescue others without extreme markup
- deception_rate: probability of taking a misleading but not strictly best-response action
- info_decay: how quickly old observations are discounted
## 10. More detailed strategy notes
Some show-relevant ideas that would be fun if there is time:
### Pure Drawer / Trusted Triangle
In a simulator, this family should search for small trusted subgraphs that can mutually exhaust cards while preserving stars. Key variables: alliance size, willingness to verify hands, and tolerance for slight asymmetry.
### Scarcity Cornerer
This family treats the floor like a commodity market. If Paper becomes scarce, the agent may hold Paper not only because it wins against Rock, but because other players may literally need Paper to avoid getting trapped. The payoff can come from either match advantage or from later trade leverage.
### Shark Seller
This strategy is only fully meaningful in Version C. It intentionally seeks extra stars, finishes early, then waits for panic. It should evaluate both price and counterparty desperation, because the best sale is not always to the first buyer.
### Debt Minimizer
Canon debt pressure makes finishing early valuable. This family may avoid high-upside but slow predatory lines, preferring reliable completion, quick exits, and modest star margins over elaborate traps.
## 11. Recommended experiment structure
Because performance depends heavily on population mix, evaluate strategies in populations rather than only in isolated mirror matches. A strong predator may look brilliant in a timid population and terrible in an all-predator pool. Likewise, a loyal coalition strategy may fail in random populations but dominate once honest subgroups are common.
- Population sweeps: test mixtures such as 40% Survivors, 20% Brokers, 20% Predators, 20% Noise players.
- Ablations: run with and without public total-card counts; with and without trading; with and without endgame market.
- Robustness tests: evaluate each strategy against several population distributions rather than one benchmark pool.
- Sensitivity analysis: vary time pressure, interest rate, market liquidity, and information visibility.
- Self-play + ecology: let population shares evolve across generations and see which personality families become dominant.
| Experiment question | Suggested setup |
| --- | --- |
| Does public card-total information change the meta? | Compare Version B with public remaining-card counts on vs. off. |
| How valuable is trust? | Run repeated populations with stable IDs and reputation memory vs. anonymous one-shot interactions. |
| How profitable is predation? | Use Version C with aggressive endgame market and compare predator vs. broker returns. |
| Does early borrowing help? | Sweep initial borrow sizes and check net wealth after interest and market outcomes. |
| Which strategies are robust? | Score across many mixed populations and rank by average + worst-case percentile. |

## 12. Potential API+state structure for simulator
```json
{
  "time_step": 37,
  "phase": "negotiation",
  "self": {
    "cards": {"R": 2, "P": 5, "S": 1},
    "stars": 4,
    "money": 3200000,
    "debt_due": 4460000,
    "status": "available"
  },
  "public": {
    "remaining_cards": {"R": 41, "P": 27, "S": 19},
    "active_players": 13,
    "recent_sales": [1.8, 2.1, 2.4]
  },
  "visible_players": [
    {"id": 7, "seen_plays": {"R": 3, "P": 0, "S": 4}, "stars": 2, "trust": 0.15},
    {"id": 12, "seen_plays": {"R": 1, "P": 3, "S": 1}, "stars": 5, "trust": 0.84}
  ],
  "inbox": {
    "match_requests": [12],
    "trade_offers": [{"from": 7, "offer": "buy_star", "price": 2.6}]
  }
}
```
Core action types:
- request_match(target_id)
- accept_match(target_id) / refuse_match(target_id)
- play_card(card_type)
- propose_trade(target_id, package)
- accept_trade(offer_id) / reject_trade(offer_id)
- post_bid(price, quantity) / post_ask(price, quantity)
- share_info(target_id, payload) / misreport_info(target_id, payload)
- wait()
## 13. Important implementation toggles

| Toggle | Examples of values | Why it matters |
| --- | --- | --- |
| Observation scope | global_public / local_only / gossip | Controls how inference-heavy the game becomes. |
| Trading allowed | none / cards_only / stars_only / both | Huge effect on coalition and market strategies. |
| Identity persistence | anonymous / stable_ids | Stable IDs make trust and revenge meaningful. |
| Cooldown length | 0, 1, 2, ... steps | Affects hunting and liquidity. |
| Interest model | none / discrete_compound / continuous_approx | Changes urgency in Version C. |
| Endgame market | off / fixed_price / bilateral bargaining / order book | Determines value of surplus stars. |
| Finish behavior | leave_immediately / seller_pool / spectator_seller | Changes post-finish strategy space. |
| Failure timing | immediate on star-bust / only at terminal check | Affects desperate tactical lines. |

## 14. References
- [C1] Kaiji Wiki, “Episode 2: Open Fire.” Summary/plot lines describing the round procedure, simultaneous face-down reveal, draw rule, and win condition.
- [C2] Kaiji Wiki, “Gyakkyō Burai Kaiji: Ultimate Survivor.” Synopsis lines describing 3 stars, 4 cards of each type, four-hour limit, and 1.5% interest every 10 minutes (noting the story rounds the total to ~40%). Additionally, the ‘War fund loans’ are loans that players were required to take from the company of between ¥1,000,000 and ¥10,000,000, which is important when the purchasing of stars and cards is considered.
- [C3] Kaiji Wiki, “Restricted Rock, Paper, Scissors.” Gameplay/aftermath lines describing surplus stars, cash-out, and the short buy/sell period for players short on stars.
- [C4] Wikipedia, “Kaiji (manga)” — Gambles / Series 1 summary for cross-checking the 50% survival framing, twelve cards, three stars, and loan structure.
- [C5] Wikipedia, “Kaiji: Ultimate Survivor” episode summaries covering team formation, card reshuffling, buying out Paper cards, and late-game information leakage/trade pressure.
## 15. Example objective profiles
| Profile | Illustrative terminal score |
| --- | --- |
| Survival-first | 100·Survival + 2·max(FinalStars-3, 0) - 0.5·TimeStepsUsed |
| Star broker | 80·Survival + 10·MarketPnL + 3·FinalStars - 1·Debt |
| Predator | 70·Survival + 6·FinalStars + 4·ExploitYield + 2·NetCash - 1·Debt |
| Debt minimizer | 90·Survival - 3·Debt + 2·FinishEarlyBonus |
| Loyalist | 85·Survival + 4·AllianceSuccess + 2·Reputation - 5·Betrayals |
