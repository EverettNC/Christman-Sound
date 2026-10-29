"""The transition model at the head of final.mdl.

It maps each transition id to its pdf (the network output column), its phone,
whether it is a self-loop, and whether it enters the topology's final state.
Transition ids start at 1; index 0 of every array is unused.
"""

from __future__ import annotations

import numpy as np

from ..modelfile import FormatError, Reader


class Topology:
    """The HMM topology: per phone, a list of states, each with transitions."""

    def __init__(self, r: Reader) -> None:
        r.expect("<Topology>")
        self.phones = r.int_vector()
        self.phone2idx = r.int_vector()
        n = r.int32()
        is_hmm = True
        if n == -1:
            is_hmm = False
            n = r.int32()
        # entries[i][s] = (forward_pdf_class, self_loop_pdf_class, [(dest, prob), ...])
        self.entries: list[list[tuple[int, int, list[tuple[int, float]]]]] = []
        for _ in range(n):
            states = []
            for _ in range(r.int32()):
                fwd = r.int32()
                loop = fwd if is_hmm else r.int32()
                trans = []
                for _ in range(r.int32()):
                    dest = r.int32()
                    prob = r.float32()
                    trans.append((dest, prob))
                states.append((fwd, loop, trans))
            self.entries.append(states)
        r.expect("</Topology>")

    def for_phone(self, phone: int):
        if phone < 0 or phone >= len(self.phone2idx) or self.phone2idx[phone] < 0:
            raise FormatError(f"phone {phone} has no topology")
        return self.entries[self.phone2idx[phone]]


class TransitionModel:
    """Transition ids and what each one means.

    Arrays are indexed by transition id, 1 .. num_tids; index 0 is unused:
      tid2pdf, tid2phone, tid2state (hmm state), tid_is_selfloop, tid_is_final,
      tid_logprob (the stored transition log probability).
    """

    def __init__(self, r: Reader) -> None:
        r.expect("<TransitionModel>")
        self.topology = Topology(r)
        tok = r.token()
        if tok not in ("<Triples>", "<Tuples>"):
            raise FormatError(f"transition model: expected <Triples> or <Tuples>, got {tok!r}")
        n = r.int32()
        tuples = np.zeros((n, 4), dtype=np.int32)  # phone, hmm_state, forward_pdf, self_loop_pdf
        for i in range(n):
            phone = r.int32()
            hmm_state = r.int32()
            fwd = r.int32()
            loop = r.int32() if tok == "<Tuples>" else fwd
            tuples[i] = (phone, hmm_state, fwd, loop)
        r.expect("</" + tok[1:])
        r.expect("<LogProbs>")
        log_probs = r.vector()
        r.expect("</LogProbs>")
        r.expect("</TransitionModel>")
        self.tuples = tuples

        topo = self.topology
        counts = np.zeros(n, dtype=np.int64)
        for i, (phone, hs, _, _) in enumerate(tuples):
            counts[i] = len(topo.for_phone(int(phone))[hs][2])
        num_tids = int(counts.sum())
        size = num_tids + 1
        self.num_tids = num_tids
        self.num_pdfs = int(max(tuples[:, 2].max(), tuples[:, 3].max())) + 1 if n else 0
        self.tid2pdf = np.zeros(size, dtype=np.int32)
        self.tid2phone = np.zeros(size, dtype=np.int32)
        self.tid2state = np.zeros(size, dtype=np.int32)
        self.tid2tstate = np.zeros(size, dtype=np.int32)  # 1-based tuple index
        self.tid_is_selfloop = np.zeros(size, dtype=bool)
        self.tid_is_final = np.zeros(size, dtype=bool)
        tid = 1
        for i, (phone, hs, fwd, loop) in enumerate(tuples):
            states = topo.for_phone(int(phone))
            final_state = len(states) - 1
            for dest, _ in states[hs][2]:
                is_loop = dest == hs
                self.tid2pdf[tid] = loop if is_loop else fwd
                self.tid2phone[tid] = phone
                self.tid2state[tid] = hs
                self.tid2tstate[tid] = i + 1
                self.tid_is_selfloop[tid] = is_loop
                self.tid_is_final[tid] = dest == final_state
                tid += 1
        if len(log_probs) != size:
            raise FormatError(f"transition model: {len(log_probs)} log probs for {num_tids} transition ids")
        self.tid_logprob = log_probs.astype(np.float32)
