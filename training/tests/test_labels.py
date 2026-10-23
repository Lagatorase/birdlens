import pytest

from birdlens_train.labels import clean_label


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("001.Black_footed_Albatross", "Black-footed Albatross"),
        ("013.Bobolink", "Bobolink"),
        ("026.Bronzed_Cowbird", "Bronzed Cowbird"),
        ("049.Boat_tailed_Grackle", "Boat-tailed Grackle"),
        ("101.White_Pelican", "White Pelican"),
        ("190.Red_cockaded_Woodpecker", "Red-cockaded Woodpecker"),
        ("200.Common_Yellowthroat", "Common Yellowthroat"),
        # These two need the special-case table.
        ("022.Chuck_will_Widow", "Chuck-will's-widow"),
        ("024.Red_faced_Cormorant", "Red-faced Cormorant"),
        ("023.Brandt_Cormorant", "Brandt's Cormorant"),
    ],
)
def test_clean_label(raw, expected):
    assert clean_label(raw) == expected


def test_clean_label_without_prefix_and_whitespace():
    # I make sure stray whitespace and a missing prefix don't break it.
    assert clean_label("  Black_footed_Albatross\n") == "Black-footed Albatross"
