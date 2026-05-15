# Recording the demo GIF

The README references `assets/scorecard_demo.gif`. Producing it is a manual step because terminal recording tools aren't available in CI.

## Recommended tool: `vhs` (Charm)

Install: `go install github.com/charmbracelet/vhs@latest` (requires Go), or grab the binary release from https://github.com/charmbracelet/vhs/releases.

`scorecard_demo.tape`:

```vhs
Output assets/scorecard_demo.gif

Set FontSize 14
Set Width 1100
Set Height 700
Set Theme "Dracula"
Set TypingSpeed 30ms

# Set the scene: about to audit a strategy
Type "# Before you trust an LLM trading agent, audit it."
Sleep 1.5s
Enter

Type "abl evaluate buy_and_hold --window 2020-01-06..2024-12-31 --out demo_out/"
Sleep 600ms
Enter
Sleep 4s

Type "head -45 demo_out/report.md"
Enter
Sleep 5s

# Now run the deliberately-overfit fixture and watch the flag fire
Type "# Now run the deliberately-overfit fixture — watch the flag fire."
Sleep 1.5s
Enter

# (placeholder for an overfit-fixture CLI route; this script is a sketch — wire it up in 0.2)
```

Produce the GIF:

```bash
vhs assets/scorecard_demo.tape
```

The GIF should be ≤ 1.5 MB (GitHub renders inline up to 5 MB, but smaller is better for first-paint).

## Alternative: asciinema + agg

```bash
asciinema rec scorecard_demo.cast
# do the same sequence above; Ctrl-D to stop
agg --theme monokai scorecard_demo.cast assets/scorecard_demo.gif
```

## Manual fallback

If neither tool is available: a screencast with QuickTime / Windows Game Bar, exported to MP4, then converted to GIF with ffmpeg:

```bash
ffmpeg -i input.mp4 -vf "fps=12,scale=1100:-1:flags=lanczos" -loop 0 assets/scorecard_demo.gif
```

## Editorial guidance

- Lead with the *problem*, not the tool. The first frame should be a strategy that *looks* great.
- Reveal a flag in the second beat. The viewer should see the banner flip color.
- Cap at ~20 seconds; under 5 MB.
- The disclaimer line MUST be visible in the recorded terminal at least once.
