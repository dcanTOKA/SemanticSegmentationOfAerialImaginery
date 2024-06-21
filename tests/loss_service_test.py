import pytest
import torch

from services.loss_service import LossService

_ = torch.manual_seed(0)


@pytest.fixture
def tensor_fixture():
    y_true = torch.randint(0, 2, (10, 3, 128, 128))
    y_pred = torch.randint(0, 2, (10, 3, 128, 128))

    return y_true, y_pred


def test_iou(tensor_fixture):
    service = LossService()
    y_true, y_pred = tensor_fixture
    result = service.iou_per_class(y_true, y_pred)
    expected = [0.3322, 0.3303, 0.3329]
    assert torch.isclose(result, torch.tensor(expected), atol=1e-4).all()
