from unittest.mock import Mock

import pytest

from qibolab._core.instruments.qblox.cluster import Cluster


def test_disconnect_closes_cluster_when_stopping_sequencer_fails():
    cluster = Cluster(name="disconnect_test", address="dummy", channels={})
    failing = Mock()
    failing.stop_sequencer.side_effect = RuntimeError("stop failed")
    other = Mock()
    connection = Mock()
    cluster._cluster = connection
    cluster.__dict__["_modules"] = {7: failing, 10: other}

    with pytest.raises(ExceptionGroup, match="disconnecting Qblox cluster"):
        cluster.disconnect()

    # every module is stopped, and the connection closed, despite the failure
    failing.stop_sequencer.assert_called_once_with()
    other.stop_sequencer.assert_called_once_with()
    connection.close.assert_called_once_with()
    assert cluster._cluster is None
    assert "_modules" not in cluster.__dict__
