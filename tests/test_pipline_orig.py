from calm.alphabet import Alphabet
from calm.sequence import CodonSequence
from calm.pipeline_original import (
    PipelineCfg,
    Pipeline,
    PipelineInput,
    MaskAndChange,
    DataTrimmer,
    DataPadder,
    DataPreprocessor,
)


def test_DataCollator_codon():
    cfg = PipelineCfg()
    alphabet = Alphabet.from_architecture("CodonModel")
    data_collator = MaskAndChange(cfg, alphabet.coding_toks)

    seq1 = CodonSequence("AUG GGA CGC UUU UAC CAA AUG GGA CGC UUU UAC CAA UAA " * 10)
    seq2 = CodonSequence("AUG GGA CGC UAA")
    input_ = PipelineInput(sequences=[seq1, seq2])
    output = data_collator(input_)

    assert output.ground_truth[0] == seq1.seq
    assert output.sequence[0].split().count("<mask>") == int(
        len(seq1.tokens) * cfg.mask_percent * cfg.mask_proportion
    )
    s = int(output.target_mask[0].sum())
    d = int(len(seq1.tokens) * cfg.mask_proportion)
    assert s == d, (s, d)


def test_DataTrimmer_codon():
    args = PipelineCfg()
    alphabet = Alphabet.from_architecture("CodonModel")
    data_trimmer = Pipeline(
        [MaskAndChange(args, alphabet.coding_toks), DataTrimmer(args.max_positions)]
    )

    seq1 = CodonSequence("AUG GGA CGC UUU UAC CAA AUG GGA CGC UUU UAC CAA UAA " * 10)
    seq2 = CodonSequence("AUG GGA CGC UAA")
    output = data_trimmer([seq1, seq2])


def test_DataPadder_codon():
    args = PipelineCfg()
    alphabet = Alphabet.from_architecture("CodonModel")
    data_padder = Pipeline(
        [
            MaskAndChange(args, alphabet.coding_toks),
            DataTrimmer(args.max_positions),
            DataPadder(args.max_positions),
        ]
    )

    seq1 = CodonSequence("AUG GGA CGC UUU UAC CAA AUG GGA CGC UUU UAC CAA UAA " * 10)
    seq2 = CodonSequence("AUG GGA CGC UAA")
    output = data_padder([seq1, seq2])


def test_DataPreprocessor_codon():
    args = PipelineCfg()
    alphabet = Alphabet.from_architecture("CodonModel")
    data_preprocessor = Pipeline(
        [
            MaskAndChange(args, alphabet.coding_toks),
            DataTrimmer(args.max_positions),
            DataPadder(args.max_positions),
            DataPreprocessor(alphabet),
        ]
    )

    seq1 = CodonSequence("AUG GGA CGC UUU UAC CAA AUG GGA CGC UUU UAC CAA UAA " * 10)
    seq2 = CodonSequence("AUG GGA CGC UAA")
    output = data_preprocessor([seq1, seq2])


test_DataCollator_codon()
