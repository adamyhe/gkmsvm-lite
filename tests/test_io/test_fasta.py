import pytest

from gkmsvm.fasta import read_fasta, write_fasta


class TestReadFasta:
    def test_basic(self, tmp_path):
        p = tmp_path / "test.fa"
        p.write_text(">seq1\nACGTACGT\n>seq2\nTGCATGCA\n")
        records = read_fasta(p)
        assert len(records) == 2
        assert records[0] == ("seq1", "ACGTACGT")
        assert records[1] == ("seq2", "TGCATGCA")

    def test_multiline_sequence(self, tmp_path):
        p = tmp_path / "multi.fa"
        p.write_text(">seq1\nACGT\nACGT\nACGT\n")
        records = read_fasta(p)
        assert records[0] == ("seq1", "ACGTACGTACGT")

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.fa"
        p.write_text("")
        records = read_fasta(p)
        assert records == []

    def test_blank_lines(self, tmp_path):
        p = tmp_path / "blanks.fa"
        p.write_text(">seq1\nACGT\n\n>seq2\nTGCA\n\n")
        records = read_fasta(p)
        assert len(records) == 2

    def test_header_with_description(self, tmp_path):
        p = tmp_path / "desc.fa"
        p.write_text(">seq1 some description here\nACGT\n")
        records = read_fasta(p)
        assert records[0][0] == "seq1 some description here"


class TestWriteFasta:
    def test_roundtrip(self, tmp_path):
        records = [("seq1", "ACGTACGT"), ("seq2", "TGCATGCA")]
        p = tmp_path / "out.fa"
        write_fasta(records, p)
        result = read_fasta(p)
        assert result == records

    def test_line_wrapping(self, tmp_path):
        seq = "A" * 200
        records = [("long", seq)]
        p = tmp_path / "wrap.fa"
        write_fasta(records, p, line_width=80)
        text = p.read_text()
        lines = text.strip().split("\n")
        assert lines[0] == ">long"
        assert len(lines[1]) == 80
        assert len(lines[2]) == 80
        assert len(lines[3]) == 40
        result = read_fasta(p)
        assert result[0][1] == seq

    def test_custom_line_width(self, tmp_path):
        records = [("seq", "ACGTACGTACGT")]
        p = tmp_path / "narrow.fa"
        write_fasta(records, p, line_width=4)
        text = p.read_text()
        lines = text.strip().split("\n")
        assert lines[1:] == ["ACGT", "ACGT", "ACGT"]


class TestExtractLoci:
    @pytest.fixture
    def genome(self, tmp_path):
        p = tmp_path / "genome.fa"
        p.write_text(">chr1\nACGTACGTAC\nGTACGTACGT\n>chr2\nNNACGTNNAC\n")
        return p

    def test_mask_marks_dropped_loci(self, genome):
        pd = pytest.importorskip("pandas")
        from gkmsvm.fasta import extract_loci

        loci = pd.DataFrame(
            {
                "chrom": ["chr1", "chr1", "chr1", "chr3", "chr2"],
                "start": [4, 0, 18, 4, 4],
                "end": [6, 2, 20, 6, 6],
            }
        )
        X, mask = extract_loci(loci, genome, in_window=4, return_mask=True)
        # Second and third run off chr1, fourth is not in the genome.
        assert mask.tolist() == [True, False, False, False, True]
        assert X.shape == (2, 4, 4)
        assert X[0].argmax(axis=0).tolist() == [3, 0, 1, 2]  # TACG
        assert X[1].sum(axis=0).tolist() == [1, 1, 1, 0]  # CGTN

    def test_mask_covers_chroms_filter(self, genome):
        pd = pytest.importorskip("pandas")
        from gkmsvm.fasta import extract_loci

        loci = pd.DataFrame(
            {
                "chrom": ["chr2", "chr1"],
                "start": [4, 4],
                "end": [6, 6],
            }
        )
        X, mask = extract_loci(
            loci, genome, in_window=4, chroms=["chr1"], return_mask=True
        )
        assert mask.tolist() == [False, True]
        assert len(loci[mask]) == len(X) == 1

    def test_default_returns_sequences_only(self, genome):
        pd = pytest.importorskip("pandas")
        from gkmsvm.fasta import extract_loci

        loci = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [2]})
        out = extract_loci(loci, genome, in_window=4)
        assert len(out) == 1
        assert out[0].shape == (0, 4, 4)
