import unittest
from unittest.mock import patch
from google.genai import types
from demo_grounded_search import GroundedResult, _build_config, _sdk_response_snapshot, parse_grounding, grounding_metadata_to_dict
from web_app import app


class GroundingParameterTests(unittest.TestCase):
    def test_rag_configuration_is_sent(self):
        c = _build_config(mode='rag_engine', rag_corpus='projects/p/locations/us/ragCorpora/c',
            rag_top_k=7, rag_distance=0.5, rag_metadata_filter='author == "Alice"',
            rag_file_ids='one, two', rag_hybrid_alpha=0.3,
            rag_ranker_type='rank_service', rag_ranker_model='semantic-ranker-512@latest',
            temperature=0.4, max_output_tokens=300, retrieval_language='zh-CN')
        store = c.tools[0].retrieval.vertex_rag_store
        self.assertEqual(store.rag_resources[0].rag_file_ids, ['one', 'two'])
        self.assertEqual(store.rag_retrieval_config.top_k, 7)
        self.assertEqual(store.rag_retrieval_config.filter.vector_distance_threshold, 0.5)
        self.assertEqual(store.rag_retrieval_config.filter.metadata_filter, 'author == "Alice"')
        self.assertEqual(store.rag_retrieval_config.hybrid_search.alpha, 0.3)
        self.assertEqual(store.rag_retrieval_config.ranking.rank_service.model_name, 'semantic-ranker-512@latest')
        self.assertEqual(c.tool_config.retrieval_config.language_code, 'zh-CN')
        self.assertEqual(c.temperature, 0.4)
        self.assertEqual(c.max_output_tokens, 300)

    def test_search_maps_and_agent_options(self):
        search = _build_config(mode='search', exclude_domains=['example.com'])
        self.assertEqual(search.tools[0].google_search.exclude_domains, ['example.com'])
        maps = _build_config(mode='maps', maps_enable_widget=True)
        self.assertTrue(maps.tools[0].google_maps.enable_widget)
        agent = _build_config(mode='agent_search', agent_search_datastore='projects/p/locations/global/collections/default_collection/dataStores/x', agent_max_results=4, agent_filter='category: ANY("a")')
        self.assertEqual(agent.tools[0].retrieval.vertex_ai_search.max_results, 4)
        self.assertEqual(agent.tools[0].retrieval.vertex_ai_search.filter, 'category: ANY("a")')
        engine = _build_config(mode='agent_search', agent_engine='projects/p/locations/global/collections/default_collection/engines/e')
        self.assertTrue(engine.tools[0].retrieval.vertex_ai_search.engine.endswith('/engines/e'))
        elastic = _build_config(mode='elasticsearch', external_endpoint='https://example.com', external_api_key='test', elastic_index='i', elastic_template='t', elastic_num_hits=5)
        self.assertEqual(elastic.tools[0].retrieval.external_api.elastic_search_params.num_hits, 5)
        search = _build_config(mode='search', search_blocking_confidence='BLOCK_HIGH_AND_ABOVE')
        self.assertEqual(search.tools[0].google_search.blocking_confidence.value, 'BLOCK_HIGH_AND_ABOVE')

    def test_form_invalid_mutually_exclusive_thresholds(self):
        c = app.test_client()
        response = c.post('/ask/stream', data={'question':'测试','mode':'rag_engine','rag_distance':'0.4','rag_similarity':'0.7'})
        self.assertEqual(response.status_code, 400)
        self.assertIn('只能填写一个', response.get_json()['error'])
        response = c.post('/count-tokens', data={'question':'测试','mode':'agent_search','agent_max_results':'11'})
        self.assertEqual(response.status_code, 400)

    def test_form_values_propagate_to_regular_stream_and_count(self):
        client = app.test_client()
        form = {'question':'测试', 'mode':'rag_engine','rag_corpus':'projects/p/locations/us/ragCorpora/c', 'rag_top_k':'4', 'rag_distance':'0.55', 'temperature':'0.4'}
        with patch('web_app.get_client', return_value=object()), patch('web_app.grounded_ask', return_value=GroundedResult(status='ungrounded', text='未找到')) as ask:
            response = client.post('/ask', data=form)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(ask.call_args.kwargs['rag_top_k'], 4)
            self.assertEqual(ask.call_args.kwargs['rag_distance'], 0.55)
            self.assertEqual(ask.call_args.kwargs['temperature'], 0.4)
        with patch('web_app.get_client', return_value=object()), patch('web_app.grounded_ask_stream', return_value=iter([{'event':'done','data':{'status':'ungrounded'}}])) as stream:
            self.assertEqual(client.post('/ask/stream', data=form).status_code, 200)
            self.assertEqual(stream.call_args.kwargs['rag_top_k'], 4)
        with patch('web_app.get_client', return_value=object()), patch('web_app.count_tokens_for_request', return_value={'input_tokens':12}) as count:
            self.assertEqual(client.post('/count-tokens', data=form).status_code, 200)
            self.assertEqual(count.call_args.kwargs['rag_top_k'], 4)

    def test_sdk_metadata_snapshot_retains_fields(self):
        gm = types.GroundingMetadata(web_search_queries=['中文查询'], grounding_chunks=[types.GroundingChunk(web=types.GroundingChunkWeb(uri='https://example.com', title='资料'))])
        response = types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(parts=[types.Part(text='回答')]), grounding_metadata=gm)])
        result = parse_grounding(response, model='test', mode='search')
        self.assertEqual(result.grounding_metadata_api['webSearchQueries'], ['中文查询'])
        self.assertEqual(grounding_metadata_to_dict(gm)['sdk_grounding_metadata']['groundingChunks'][0]['web']['title'], '资料')

    def test_api_reference_and_optional_defaults(self):
        response = app.test_client().get('/api-reference')
        self.assertEqual(response.status_code, 200)
        self.assertIn('vector_distance_threshold', response.get_data(as_text=True))
        store = _build_config(mode='rag_engine', rag_corpus='projects/p/locations/us/ragCorpora/c').tools[0].retrieval.vertex_rag_store
        self.assertIsNone(store.rag_retrieval_config)

    def test_response_snapshot_omits_inline_binary(self):
        response = types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(parts=[types.Part(inline_data=types.Blob(mime_type='image/png', data=b'fake-image'))]))])
        snapshot = _sdk_response_snapshot(response)
        self.assertIn('已省略', snapshot['candidates'][0]['content']['parts'][0]['inlineData']['data'])

    def test_empty_grounding_metadata_is_not_described_as_evidence(self):
        response = types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(parts=[types.Part(text='回答')]), grounding_metadata=types.GroundingMetadata())])
        result = parse_grounding(response, model='test', mode='rag_engine')
        self.assertIn('没有可展示的来源或引用', result.warning)

    def test_all_modes_build_with_sdk(self):
        cases = {
            'search': {}, 'image_search': {}, 'maps': {}, 'url_context': {},
            'agent_search': {'agent_search_datastore': 'projects/p/locations/global/collections/default_collection/dataStores/d'},
            'rag_engine': {'rag_corpus': 'projects/p/locations/us/ragCorpora/c'},
            'external_search': {'external_endpoint': 'https://example.com', 'external_api_key': 'test'},
            'elasticsearch': {'external_endpoint': 'https://example.com', 'external_api_key': 'test', 'elastic_index': 'index', 'elastic_template': 'template'},
            'parallel_web': {}, 'enterprise_web': {},
        }
        for mode, options in cases.items():
            with self.subTest(mode=mode):
                self.assertTrue(_build_config(mode=mode, **options).tools)

if __name__ == '__main__':
    unittest.main()
