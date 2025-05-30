import pytest
from src.chatbot import doctor_gpt

def test_doctor_gpt_returns_string(mocker):
    """
    Test that doctor_gpt returns a string response for a health condition query.
    """
    user_query = "What is Candidiasis?"

    # Mock the external dependencies
    mock_search_db = mocker.patch('src.database.search_db')
    mock_doctor_gpt_ai = mocker.patch('src.chatbot.doctor_gpt_ai')

    # Define return values for the mocks
    mock_search_db.return_value = [
        'Candidiasis\nThis patient’s tongue is infected with candidiasis.(Photo-\ngraph by Edward H. Gill, Custom Medical Stock Photo. Repro-\nduced by permission.)\nGEM -0625 to 1002 - C  10/22/03 6:10 PM  Page 646',
        'teristic forms of yeasts at various stages in the lifecycle.\nFungal blood cultures should be taken for patients\nsuspected of having deep organ candidiasis. Tissue biop-\nsy may be needed for a definitive diagnosis.\nTreatment\nVaginal candidiasis\nIn most cases, vaginal candidiasis can be treated\nsuccessfully with a variety of over-the-counter antifungal\ncreams or suppositories. These include Monistat, Gyne-\nLotrimin, and Mycelex. However, infections often recur.',
        'of which increase a patient’s susceptibility to infection.\nDescription\nVaginal candidiasis\nOver one million women in the United States devel-\nop vaginal yeast infections each year. It is not life-threat-\nening, but it can be uncomfortable and frustrating.\nOral candidiasis\nThis disorder, also known as thrush, causes white,\ncurd-like patches in the mouth or throat.\nGALE ENCYCLOPEDIA OF MEDICINE 2 645\nCandidiasis\nGEM -0625 to 1002 - C  10/22/03 6:10 PM  Page 645'
    ]
    mock_doctor_gpt_ai.return_value = (
        "Candidiasis is an infection caused by yeast, with various forms, including vaginal candidiasis, "
        "which affects over one million women in the United States each year, and oral candidiasis, also "
        "known as thrush, which causes white, curd-like patches in the mouth or throat. It is not life-threatening "
        "but can be uncomfortable and frustrating. Treatment for vaginal candidiasis often involves over-the-counter "
        "antifungal creams or suppositories, such as Monistat, Gynelotrimin, and Mycelex, while deep organ candidiasis "
        "may require fungal blood cultures and tissue biopsy for a definitive diagnosis."
    )

    # Call the function
    response = doctor_gpt(user_query)

    # Assertions
    mock_search_db.assert_called_once_with(user_query=user_query)
    mock_doctor_gpt_ai.assert_called_once_with(
        user_query=user_query,
        doc_list=[
            'Candidiasis\nThis patient’s tongue is infected with candidiasis.(Photo-\ngraph by Edward H. Gill, Custom Medical Stock Photo. Repro-\nduced by permission.)\nGEM -0625 to 1002 - C  10/22/03 6:10 PM  Page 646',
            'teristic forms of yeasts at various stages in the lifecycle.\nFungal blood cultures should be taken for patients\nsuspected of having deep organ candidiasis. Tissue biop-\nsy may be needed for a definitive diagnosis.\nTreatment\nVaginal candidiasis\nIn most cases, vaginal candidiasis can be treated\nsuccessfully with a variety of over-the-counter antifungal\ncreams or suppositories. These include Monistat, Gyne-\nLotrimin, and Mycelex. However, infections often recur.',
            'of which increase a patient’s susceptibility to infection.\nDescription\nVaginal candidiasis\nOver one million women in the United States devel-\nop vaginal yeast infections each year. It is not life-threat-\nening, but it can be uncomfortable and frustrating.\nOral candidiasis\nThis disorder, also known as thrush, causes white,\ncurd-like patches in the mouth or throat.\nGALE ENCYCLOPEDIA OF MEDICINE 2 645\nCandidiasis\nGEM -0625 to 1002 - C  10/22/03 6:10 PM  Page 645'
        ]
    )
    assert isinstance(response, str)
    assert response == (
        "Candidiasis is an infection caused by yeast, with various forms, including vaginal candidiasis, "
        "which affects over one million women in the United States each year, and oral candidiasis, also "
        "known as thrush, which causes white, curd-like patches in the mouth or throat. It is not life-threatening "
        "but can be uncomfortable and frustrating. Treatment for vaginal candidiasis often involves over-the-counter "
        "antifungal creams or suppositories, such as Monistat, Gynelotrimin, and Mycelex, while deep organ candidiasis "
        "may require fungal blood cultures and tissue biopsy for a definitive diagnosis."
    )

def test_doctor_gpt_empty_query(mocker):
    """
    Test doctor_gpt behavior with an empty health condition query.
    """
    user_query = ""

    mock_search_db = mocker.patch('src.database.search_db')
    mock_doctor_gpt_ai = mocker.patch('src.chatbot.doctor_gpt_ai')

    mock_search_db.return_value = []
    mock_doctor_gpt_ai.return_value = "Please provide a valid health condition query."

    response = doctor_gpt(user_query)

    mock_search_db.assert_called_once_with(user_query=user_query)
    mock_doctor_gpt_ai.assert_called_once_with(user_query=user_query, doc_list=[])
    assert response == "Please provide a valid health condition query."