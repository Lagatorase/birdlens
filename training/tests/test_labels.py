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
        ("024.Red_faced_Cormorant", "Red-faced Cormorant"),
        # Hyphen chains with several lowercase parts.
        ("159.Black_and_white_Warbler", "Black-and-white Warbler"),
        ("163.Cape_May_Warbler", "Cape May Warbler"),
        # These need the special-case table for the missing apostrophe.
        ("022.Chuck_will_Widow", "Chuck-will's-widow"),
        ("023.Brandt_Cormorant", "Brandt's Cormorant"),
        ("067.Anna_Hummingbird", "Anna's Hummingbird"),
        ("126.Nelson_Sharp_tailed_Sparrow", "Nelson's Sharp-tailed Sparrow"),
        ("146.Forsters_Tern", "Forster's Tern"),
        ("105.Whip_poor_Will", "Whip-poor-will"),
        ("141.Artic_Tern", "Arctic Tern"),
    ],
)
def test_clean_label(raw, expected):
    assert clean_label(raw) == expected


def test_clean_label_without_prefix_and_whitespace():
    # I make sure stray whitespace and a missing prefix don't break it.
    assert clean_label("  Black_footed_Albatross\n") == "Black-footed Albatross"
