"""Journal: notes, tags and a star rating for any paper trade, plus what the tags say."""

import pandas as pd
import streamlit as st

from sanjotracker import dashboard as d
from sanjotracker.charts import trade_figure

c = d.cfg()
db = d.db_key()
book = d.paper_book()

trades = book.trades()
if trades.empty:
    st.info(
        "Paper trades appear here once the paper trader has taken some.",
        title="Nothing to journal yet",
        icon=":material/edit_note:",
    )
    st.stop()

known_tags = book.all_tags()

with st.container(horizontal=True, vertical_alignment="bottom"):
    status = st.segmented_control(
        "Status", ["all", "open", "closed"], default="all", required=True, key="jr_status"
    )
    tag_filter = st.multiselect("Has tag", known_tags, key="jr_tags")
    unjournaled = st.toggle("Only without notes", key="jr_empty")

view = trades
if status != "all":
    view = view[view["status"] == status]
if tag_filter:
    view = view[view["tags"].map(lambda s: bool(set(s.split(",")) & set(tag_filter)))]
if unjournaled:
    view = view[view["notes"].str.strip() == ""]

table = view.assign(
    entry=d.to_ist(view["entry_ts"], "%d %b %H:%M"),
    result=view["r_multiple"].map(lambda r: "" if pd.isna(r) else f"{r:+.2f}R"),
)
sel = st.dataframe(
    table[
        [
            "id",
            "symbol",
            "timeframe",
            "label",
            "direction",
            "status",
            "entry",
            "result",
            "tags",
            "rating",
            "notes",
        ]
    ],
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key="jr_table",
    column_config={
        "rating": st.column_config.NumberColumn(format="%d ★"),
        "notes": st.column_config.TextColumn(width="large"),
    },
    alt="Paper trades with their journal notes, select one to edit",
)
rows = sel.selection.rows if sel else []
if not rows:
    st.caption("Select a trade to write notes, tags and a rating.")
else:
    t = view.iloc[rows[0]].to_dict()
    tid = int(t["id"])
    left, right = st.columns([2, 3])
    with left, st.form(f"jr_form_{tid}", border=True):
        st.markdown(f"**#{tid} {t['symbol']} {t['timeframe']} {t['direction']}** · {t['label']}")
        if t["status"] == "closed":
            st.caption(
                f"{t['exit_reason']} · {t['net_pnl']:+,.0f} · {t['r_multiple']:+.2f}R · "
                f"regime {t['regime_trend']}/{t['regime_market']}"
            )
        notes = st.text_area(
            "Notes",
            t["notes"],
            height=160,
            placeholder="Why it triggered, what you'd have done, what you learned…",
            key=f"jr_notes_{tid}",
        )
        current = [x for x in t["tags"].split(",") if x]
        tags = st.multiselect(
            "Tags",
            sorted(set(known_tags) | set(current)),
            default=current,
            accept_new_options=True,
            key=f"jr_tagsel_{tid}",
            help="Type to add a tag, e.g. news, gap, chased, textbook",
        )
        stars = st.feedback(
            "stars",
            default=None if pd.isna(t["rating"]) else int(t["rating"]) - 1,
            key=f"jr_stars_{tid}",
        )
        if st.form_submit_button("Save", type="primary", icon=":material/save:"):
            book.save_journal(tid, notes, tags, None if stars is None else stars + 1)
            st.toast("Saved", icon=":material/check:")
            st.rerun()
    with right:
        candles = d.candles(db, t["symbol"], t["timeframe"])
        if not candles.empty:
            st.plotly_chart(
                trade_figure(candles, t, is_nse=True),
                alt=f"Chart of paper trade {tid}",
            )

# ------------------------------------------------------------------ what the tags say
tagged = trades[(trades["status"] == "closed") & (trades["tags"] != "")]
if not tagged.empty:
    st.subheader("Results by tag")
    st.caption("Do the trades you tag the same way win or lose? (closed trades only)")
    exploded = tagged.assign(tag=tagged["tags"].str.split(",")).explode("tag")
    by_tag = (
        exploded.groupby("tag")
        .agg(
            trades=("id", "count"),
            win_rate=("net_pnl", lambda s: (s > 0).mean()),
            avg_r=("r_multiple", "mean"),
            net_pnl=("net_pnl", "sum"),
        )
        .reset_index()
        .sort_values("trades", ascending=False)
    )
    st.dataframe(
        by_tag,
        hide_index=True,
        column_config={
            "win_rate": st.column_config.NumberColumn(format="percent"),
            "avg_r": st.column_config.NumberColumn("avg R", format="%+.2f"),
            "net_pnl": st.column_config.NumberColumn(format="%+.0f"),
        },
        alt="Closed paper trades grouped by journal tag",
    )
