import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
import readonly_bridge as bridge
from readonly_worker import image_quality, atomic_json


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        bridge.SESSION.clear()
        self.app=FastAPI()
        @self.app.post('/start-inference')
        async def unsafe(): return {'unsafe_called': True}
        @self.app.post('/move-arm')
        async def move(): return {'unsafe_called': True}
        bridge.install(self.app)
        self.client=TestClient(self.app)

    def test_native_model_cannot_bypass_readonly(self):
        for ref in [r'G:\LeRobot\offline_kit\models\010000\pretrained_model','G:/LeRobot/offline_kit/models/006000/pretrained_model']:
            r=self.client.post('/start-inference',json={'policy_ref':ref})
            self.assertEqual(r.status_code,409)
        self.assertEqual(self.client.post('/start-inference',json={'policy_ref':'other'}).status_code,200)

    def test_exclusive_hardware(self):
        with patch.object(bridge,'active',return_value=True):
            self.assertEqual(self.client.post('/move-arm').status_code,409)
            self.assertEqual(self.client.get('/available-cameras').status_code,409)

    def test_request_cannot_select_execute_or_arbitrary_script(self):
        for body in [{'mode':'execute'},{'seconds':61},{'mode':'live','model_dir':'elsewhere'}]:
            self.assertEqual(self.client.post('/so101-readonly/api/start',json=body).status_code,422)

    def test_stop_only_signals_file(self):
        with tempfile.TemporaryDirectory() as d:
            bridge.SESSION.update(output=Path(d),id='test')
            with patch.object(bridge,'active',return_value=True),patch.object(bridge,'status',return_value={'active':True}):
                bridge.stop()
            self.assertTrue((Path(d)/'STOP').is_file())
        bridge.SESSION.clear()

    def test_black_camera_is_blocked_but_normal_image_is_not(self):
        black=np.full((480,640,3),7,dtype=np.uint8)
        image=np.tile(np.arange(256,dtype=np.uint8),(480,3,1)).transpose(0,2,1)
        q=image_quality({'1':black,'3':image})
        self.assertTrue(q['1']['near_black']);self.assertFalse(q['3']['near_black'])

    def test_atomic_status_and_no_arbitrary_download(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'status.json';atomic_json(path,{'status':'测试'})
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['status'],'测试')
        self.assertEqual(self.client.get('/so101-readonly/api/archive').status_code,404)

    def test_idle_page(self):
        self.assertEqual(self.client.get('/so101-readonly/api/status').json()['status'],'idle')
        self.assertIn('10k',self.client.get('/so101-readonly').text)


if __name__=='__main__':unittest.main()
