"""G2 split-query prompt (phase 06). Same MC prompt as ch.make_parts except the last part, which asks ONE question (item or country).
Importing this module patches ch.make_parts; run_models looks it up through the ch module at call time."""
import ch
_orig = ch.make_parts


def make_parts_g2(tr, mode, images=True):
    parts = _orig(tr, mode, images)
    ask = tr.get("ask")
    if not ask:
        return parts
    assert mode == "mc", "G2 is MC only"
    plural, singular = ch.CATS[tr["category"]]
    ref = f"Image {tr['slot']}"
    io = "\n".join(f"{a}. {x}" for a, x in zip(ch.LETTERS, tr["item_options"]))
    co = "\n".join(f"{a}. {x}" for a, x in zip(ch.LETTERS, tr["country_options"]))
    if ask == "item":
        q = (f"\n\nLook only at {ref} and answer one multiple-choice question.\n\n"
             f"Question. What is the name of the {singular} shown in {ref}?\n{io}\n\n"
             f"Answer in exactly this format and nothing else:\nItem: <letter>")
    else:
        q = (f"\n\nLook only at {ref} and answer one multiple-choice question.\n\n"
             f"Question. Which country is the {singular} shown in {ref} from?\n{co}\n\n"
             f"Answer in exactly this format and nothing else:\nCountry: <letter>")
    parts[-1] = ("text", q)
    return parts


ch.make_parts = make_parts_g2
