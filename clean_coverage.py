"""Drop player-seasons whose 'club' is not a real club in the covered leagues.

The data covers mainly European leagues. A player at a club outside them (MLS, Saudi Pro League,
English lower leagues, ...) only appears in national-team games, cups or qualifiers, so the club with
most minutes becomes a national team or a cup-only club: no club name, a 'squad' of a few players,
and club features built from a handful of games. Those rows are noise for training and wrong in the app.

Rule: keep a row only if the club has a name in the clubs table and at least 11 players appear for it
that season. The full table is kept as player_seasons_all.csv.
"""
import os
import pandas as pd

D = "/Users/varunvarma/Transferlens/data/processed/"
src, dst = D + "player_seasons_all.csv", D + "player_seasons.csv"
if not os.path.exists(src):
    os.rename(dst, src)
b = pd.read_csv(src, low_memory=False)
covered = b["club_name"].notna() & (b["squad_size"] >= 11)
per = pd.DataFrame({"rows": b.groupby("season").size(),
                    "dropped": (~covered).groupby(b["season"]).sum()})
per["dropped_pct"] = (100 * per["dropped"] / per["rows"]).round(1)
print(per.to_string())
print("total: kept %d, dropped %d (%.2f%%)" % (covered.sum(), (~covered).sum(), 100 * (~covered).mean()))
ex = b[~covered & (b["season"] == b["season"].max())].sort_values("mv_end", ascending=False)
print("examples dropped in %d: %s" % (b["season"].max(), ", ".join(ex["name"].head(8))))
b[covered].to_csv(dst, index=False)
print("saved", dst, "(full table kept as player_seasons_all.csv)")
