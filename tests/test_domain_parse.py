import pytest
from musk.domain_encoders import parse_domain_list

def test_parse_single_string():
    arg = "xray=/path/xray.pth,patho=/p.pth"
    assert parse_domain_list(arg) == ["xray=/path/xray.pth", "patho=/p.pth"]

def test_parse_tokens_with_spaces():
    tokens = ["xray=", "/x.pth,", "patho=", "/p.pth"]
    assert parse_domain_list(tokens) == ["xray=/x.pth", "patho=/p.pth"]
