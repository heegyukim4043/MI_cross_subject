import json
import unittest
import numpy as np
import study


class ExternalVersionTests(unittest.TestCase):
    def setUp(self):
        ext=study.module(study.OLD/'run.py','synthetic_external')
        self.op,self.ns,_=ext.load_operator(study.ROOT)
        rng=np.random.default_rng(610153)
        self.x=(rng.normal(size=(24,8,40))*np.arange(1,9)[None,:,None]*1e-6).astype(np.float32)
        self.ids=np.repeat([1,2,3],8)

    def test_full_design(self):
        cfg=json.loads((study.HOME/'config.json').read_text(encoding='utf-8'))
        self.assertEqual(len(study.cells(cfg)),8)
        self.assertFalse(cfg['target_labels_before_prediction_freeze'])
        self.assertFalse(cfg['additional_experiments_after_this'])

    def test_relative_binary_scale_invariance(self):
        for pipe in ['subject','dataset_subject']:
            a,_=self.op.align(self.x,self.ids,'relative',pipe,1.)
            b,_=self.op.align(self.x*np.float32(32),self.ids,'relative',pipe,1.)
            np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-5)

    def test_absolute_unit_contract(self):
        for pipe in ['subject','dataset_subject']:
            a,_=self.op.align(self.x,self.ids,'coherent_absolute',pipe,1.)
            b,_=self.op.align(self.x,self.ids,'coherent_absolute',pipe,1e-3)
            self.assertLess(self.op.relative_error(a,b),1e-5)

    def test_exact_whitening(self):
        a,_=self.op.align(self.x,self.ids,'relative','subject',1.)
        xx=a[self.ids==1].astype(float)
        r=np.einsum('nct,ndt->ncd',xx,xx).mean(axis=0)
        np.testing.assert_allclose(r,np.eye(8),atol=1e-6)

    def test_non_spd_rejected(self):
        with self.assertRaises(ValueError):self.op.batch_align(np.zeros_like(self.x),'relative',1e-8)

    def test_K_coefficients(self):
        from analyze import k_weights
        cfg=json.loads((study.HOME/'config.json').read_text(encoding='utf-8'))
        ids=[c['id'] for c in study.cells(cfg)]
        x=np.array([60,70,70,71,65,70,70,71])
        a,o=k_weights(ids,'adopted'),k_weights(ids,'original')
        self.assertEqual(x@a,9)
        self.assertEqual(x@o,4)
        self.assertEqual(x@(o-a),-5)

    def test_smoke_classifier(self):
        z,_=self.op.align(self.x,self.ids,'relative','subject',1.)
        mu,sd=self.ns['norm_stats'](z)
        z=((z-mu)/sd).astype(np.float32)
        csp,sc,lda=self.ns['fit_csp_lda'](self.ns['ManualCSP'],z,np.tile([0,1],12),8)
        p=lda.predict(sc.transform(csp.transform(z)))
        self.assertEqual(p.shape,(24,))


if __name__=='__main__':unittest.main()
