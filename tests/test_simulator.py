import json
import pytest
from dtrain_sim import (Example, LinearModel, NetworkConfig, ring_allreduce,
    bucketed_ring_allreduce, parameter_server_reduce, synthetic_data, train,
    sequential_step, save_checkpoint, load_checkpoint)

def close(a,b,tol=1e-10):
    assert len(a)==len(b)
    assert all(abs(x-y)<tol for x,y in zip(a,b)), (a,b)

def test_collectives_agree():
    gs=[[1.,2.,3.,4.,5.],[3.,4.,5.,6.,7.],[2.,0.,8.,4.,1.]]
    expected=[sum(g[i] for g in gs)/len(gs) for i in range(5)]
    for fn in (ring_allreduce, parameter_server_reduce):
        got,stats=fn(gs)
        close(got,expected); assert stats.messages>0 and stats.bytes_sent>0
    got,stats=bucketed_ring_allreduce(gs,2)
    close(got,expected); assert stats.messages>0

def test_ring_one_worker_has_no_network_cost():
    got,stats=ring_allreduce([[1.,2.]])
    assert got==[1.,2.] and stats.messages==stats.bytes_sent==0

def test_training_matches_sequential_full_batch():
    data=synthetic_data(120)
    initial=LinearModel([0.0],0.0); sequential=LinearModel([0.0],0.0)
    for _ in range(8): sequential_step(sequential,data,0.04)
    for strategy in ('ring','parameter-server'):
        result=train(data,initial,steps=8,learning_rate=0.04,workers=4,strategy=strategy)
        close(result.model.parameters(),sequential.parameters(),1e-9)

def test_uneven_partition_matches_sequential():
    data=synthetic_data(17); seq=LinearModel([0.],0.)
    sequential_step(seq,data,0.02)
    result=train(data,seq0:=LinearModel([0.],0.),steps=1,learning_rate=0.02,workers=4)
    close(result.model.parameters(),seq.parameters(),1e-9)

def test_bucketing_and_straggler_and_failure():
    data=synthetic_data(40)
    result=train(data,LinearModel([0.],0.),steps=5,workers=4,bucket_size=1,
                 straggler_workers={2},straggler_delay=0.25,failed_worker_steps={1:2})
    assert result.recovered_failures==1
    assert max(result.step_times)>=0.25
    assert result.losses[-1]<result.losses[0]

def test_checkpoint_round_trip(tmp_path):
    model=LinearModel([1.2,-0.5],0.3); path=tmp_path/'ckpt.json'
    save_checkpoint(path,model,7,[2.,1.])
    loaded,step,losses=load_checkpoint(path)
    assert loaded.parameters()==model.parameters() and step==7 and losses==[2.,1.]

def test_checkpoint_created_by_training(tmp_path):
    path=tmp_path/'nested'/'state.json'
    train(synthetic_data(20),LinearModel([0.],0.),steps=3,workers=2,checkpoint_path=path)
    payload=json.loads(path.read_text()); assert payload['step']==3

def test_validation():
    with pytest.raises(ValueError): ring_allreduce([])
    with pytest.raises(ValueError): bucketed_ring_allreduce([[1]],0)

def test_failure_step_uses_partial_data_honestly():
    # A failed worker's shard cannot be reconstructed by rescaling: the
    # failure-step gradient must differ from the true global mean, and must
    # equal the survivors-only rescaled mean.
    from dtrain_sim.training import partition_data
    data = synthetic_data(20)
    init = LinearModel([0.0], 0.0)
    parts = partition_data(data, 4)
    result = train(data, LinearModel([0.0], 0.0), steps=1, learning_rate=0.04,
                   workers=4, failed_worker_steps={0: 1})
    # Reproduce the survivors-only computation by hand.
    grads = []
    for i in (1, 2, 3):
        g = init.gradient(parts[i])
        grads.append([v * (len(parts[i]) * 4 / len(data)) * (3 / 4) for v in g])
    expected = [sum(g[j] for g in grads) / 3 for j in range(len(grads[0]))]
    expected_params = [0.0 - 0.04 * v for v in expected] + [0.0 - 0.04 * expected[-1]]
    # parameters() == [*weights, bias]; gradient vector already includes bias last
    close(result.model.parameters(),
          [0.0 - 0.04 * expected[0], 0.0 - 0.04 * expected[1]], 1e-9)
    # And it must NOT equal the full-data sequential step.
    seq = LinearModel([0.0], 0.0)
    sequential_step(seq, data, 0.04)
    assert result.model.parameters() != pytest.approx(seq.parameters())
