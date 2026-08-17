'use strict';

const assert = require('assert');
const os = require('os');
const path = require('path');
const fs = require('../desktop_annotation/PaperFlow-Annotation/resources/app/node_modules/fs-extra');
const {
    saveProjectCacheOnExit,
} = require('../desktop_annotation/PaperFlow-Annotation/resources/app/electron/exit-cache');

async function main() {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), 'paperflow-exit-cache-'));
    try {
        const projectPath = path.join(root, 'PaperFlow标注项目.json');
        await fs.writeJson(projectPath, { old: true });
        const largeText = '题'.repeat(100000);
        const project = {
            book_root: root,
            image_root: root,
            pages: [{ image: '001.jpg', source_path: path.join(root, '001.jpg'), questions: [{ id: 'q-1', text: largeText }] }],
            answer_pages: [],
            ignored_pages: [],
            pairs: [],
        };

        const result = await saveProjectCacheOnExit({ projectPath, project });
        assert.strictEqual(result.ok, true);
        const saved = await fs.readJson(projectPath);
        assert.strictEqual(result.project_revision, saved.project_revision);
        assert.strictEqual(saved.pages[0].questions[0].text.length, 100000);
        assert.strictEqual('source_path' in saved.pages[0], false);
        assert.ok(saved.updated_at);

        const authoritative = {
            ...saved,
            delivery_book_id: 'BBOOK123',
            pages: [{ image: '001.jpg', questions: [{ id: 'q-1', delivery_id: 'QQUESTION123', text: largeText }] }],
            answer_pages: [{ image: 'answer.jpg', answers: [{ id: 'a-1', delivery_id: 'AANSWER123' }] }],
            pairs: [{ id: 'pair-1', delivery_id: 'RRELATION123', question_id: 'q-1', answer_ids: ['a-1'], status: 'accepted' }],
        };
        await fs.writeJson(projectPath, authoritative);
        const staleProject = {
            ...project,
            project_revision: authoritative.project_revision,
            delivery_book_id: 'BSTALE',
            image_root: path.join(root, '..', 'outside'),
            answer_image_root: path.join(root, '..', 'outside-answers'),
            answer_pages: [{ image: 'answer.jpg', answers: [{ id: 'a-1', delivery_id: 'ASTALE' }] }],
            pairs: [],
        };
        await saveProjectCacheOnExit({ projectPath, project: staleProject });
        const merged = await fs.readJson(projectPath);
        assert.strictEqual(merged.project_revision, authoritative.project_revision + 1);
        assert.strictEqual(merged.delivery_book_id, 'BBOOK123');
        assert.strictEqual(merged.image_root, authoritative.image_root);
        assert.strictEqual('answer_image_root' in merged, false);
        assert.strictEqual(merged.pages[0].questions[0].delivery_id, 'QQUESTION123');
        assert.strictEqual(merged.answer_pages[0].answers[0].delivery_id, 'AANSWER123');
        assert.deepStrictEqual(merged.pairs, authoritative.pairs);

        await fs.writeJson(projectPath, authoritative);
        const missingAnswer = {
            ...staleProject,
            answer_pages: [],
            pairs: [],
        };
        await saveProjectCacheOnExit({ projectPath, project: missingAnswer });
        const withoutDanglingRelation = await fs.readJson(projectPath);
        assert.deepStrictEqual(withoutDanglingRelation.pairs, []);

        const latest = {
            ...authoritative,
            project_revision: 9,
            pages: [{ image: '001.jpg', questions: [
                { id: 'q-1', delivery_id: 'QQUESTION123' },
                { id: 'q-2', delivery_id: 'QQUESTION456' },
            ] }],
        };
        await fs.writeJson(projectPath, latest);
        await assert.rejects(
            saveProjectCacheOnExit({
                projectPath,
                project: { ...staleProject, project_revision: 8 },
            }),
            /缓存已更新/
        );
        assert.strictEqual((await fs.readJson(projectPath)).pages.length, latest.pages.length);
        assert.strictEqual((await fs.readJson(projectPath)).pages[0].questions.length, 2);

        await assert.rejects(
            saveProjectCacheOnExit({ projectPath: path.join(root, 'outside.json'), project }),
            /路径与当前教材不匹配/
        );
        assert.strictEqual((await fs.readJson(projectPath)).pages[0].questions.length, 2);
    } finally {
        await fs.remove(root);
    }
}

main().then(() => {
    process.stdout.write('exit-cache tests passed\n');
}).catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
