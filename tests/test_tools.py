"""订单/物流工具测试（模拟订单数据，不依赖任何外部服务）。"""
from src.agent.tools import MOCK_ORDERS, apply_refund, check_refund_eligibility, query_order, track_logistics

# 从模拟数据取订单号，数据变更时测试依然有效
SHIPPED = MOCK_ORDERS[0]["order_id"]   # 已发货
PENDING = MOCK_ORDERS[1]["order_id"]   # 待发货


def test_query_shipped_order():
    out = query_order.invoke({"order_id": SHIPPED})
    assert "已发货" in out
    assert MOCK_ORDERS[0]["product_name"] in out


def test_query_unknown_order():
    out = query_order.invoke({"order_id": "SO99999999999"})
    assert "未找到订单" in out


def test_track_logistics_shipped():
    out = track_logistics.invoke({"order_id": SHIPPED})
    assert MOCK_ORDERS[0]["logistics"] in out  # 含顺丰单号


def test_track_logistics_pending_has_no_tracking():
    out = track_logistics.invoke({"order_id": PENDING})
    assert "尚未发货" in out


def test_refund_eligibility_pending_order():
    out = check_refund_eligibility.invoke({"order_id": PENDING})
    assert "符合" in out


def test_refund_eligibility_shipped_order():
    out = check_refund_eligibility.invoke({"order_id": SHIPPED})
    assert "不符合" in out


def test_apply_refund():
    out = apply_refund.invoke({"order_id": PENDING, "reason": "不想要了"})
    assert "退款申请已提交" in out
    assert "1-3 个工作日" in out
