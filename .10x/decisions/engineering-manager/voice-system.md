# Engineering manager — voice-system

The order of work:
1. Start Lanes 1 to 6 in parallel. Lane 1 merges each control as its own PR, starting with breath, jitter and shimmer.
2. Lane 5 builds the pass-through reading synthesize()'s signature, the same way the VCC does, so it goes live as each Lane 1 control lands.
3. Lane 2's measured fields land in alphavox_voice.json, and Lane 1 maps them to the controls.
4. Lane 7 starts once Lanes 2 and 3 have merged.
5. QA runs once Lanes 1, 2 and 5 have merged: end to end, with fidelity and intelligibility numbers.
6. The orchestrator updates ~/Christman-Sound and ~/AlphaVox on Everett's Mac, restarts AlphaVox, and leaves samples in ~/claude-work/listen.

Risks:
- Merge order on Christman-Sound main. Every lane rebases right before it merges and runs the full suite after rebasing.
