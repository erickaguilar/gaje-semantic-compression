// =============================================================================
// merge — Fusión y migración de conocimiento entre modelos de DNIEngine
// =============================================================================
use rand::Rng;

use crate::nn::llm::GenomicLLM;

use crate::core::dni::DNIEngine;

impl DNIEngine {
    pub(crate) fn chunk_text(&self, text: &str) -> Vec<String> {
        text.lines()
            .map(|l| l.trim().to_string())
            .filter(|l| l.len() > 20)
            .collect()
    }

    pub(crate) fn merge_models(&self, base: &mut GenomicLLM, mutant: &GenomicLLM) {
        let mut rng = rand::thread_rng();
        for (i, b_base) in base.blocks.iter_mut().enumerate() {
            let b_mutant = &mutant.blocks[i];
            let layers = [
                (&mut b_base.gate_gen, &b_mutant.gate_gen),
                (&mut b_base.up_gen, &b_mutant.up_gen),
                (&mut b_base.w_down, &b_mutant.w_down),
                (&mut b_base.q_gen, &b_mutant.q_gen),
                (&mut b_base.k_gen, &b_mutant.k_gen),
                (&mut b_base.v_gen, &b_mutant.v_gen),
                (&mut b_base.w_o, &b_mutant.w_o),
            ];
            for (l_base, l_mutant) in layers {
                let db_base_vec = l_base.database_ref().to_vec();
                let db_mutant = l_mutant.database_ref();
                if db_base_vec != db_mutant {
                    let mut db_base = db_base_vec;
                    let crossover_point = rng.gen_range(0..db_base.len());
                    for j in crossover_point..db_base.len() {
                        db_base[j] = db_mutant[j];
                    }
                    l_base.database_mut().copy_from_slice(&db_base);
                }
            }
        }
    }

    pub(crate) fn migrate_knowledge(&self, logic_model: &mut GenomicLLM, grammar_model: &mut GenomicLLM) {
        let mut rng = rand::thread_rng();
        for i in 0..logic_model.blocks.len() {
            let blk_l = &mut logic_model.blocks[i];
            let blk_g = &mut grammar_model.blocks[i];

            let db_l = blk_l.w_down.database_ref().to_vec();
            let mut db_g = blk_g.w_down.database_ref().to_vec();
            for j in 0..db_l.len() {
                if rng.gen_bool(0.1) {
                    db_g[j] = db_l[j];
                }
            }
            blk_g.w_down.database_mut().copy_from_slice(&db_g);

            let mut db_attn_l = blk_l.w_o.database_ref().to_vec();
            let db_attn_g = blk_g.w_o.database_ref();
            for j in 0..db_attn_l.len() {
                if rng.gen_bool(0.1) {
                    db_attn_l[j] = db_attn_g[j];
                }
            }
            blk_l.w_o.database_mut().copy_from_slice(&db_attn_l);
        }
    }
}
