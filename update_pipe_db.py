"""Update the pipe function in SQLite with the correct signature for Open WebUI v0.10.2."""
import sqlite3

# Read the pipe code from disk
with open(r'c:\Users\ivanr\OneDrive\Documents\New folder (2)\LangChain4LifeSciencesHealthcare\legacy\lifesciencebench-v0.3.2\openwebui\pipes\lifesciencebench_research.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the pipe signature to match OWUI v0.10.2 convention
# OWUI v0.10.2 calls pipe(**params) where params = {'body': form_data} | extra_params
old_sig = (
    '    def pipe(\n'
    '        self,\n'
    '        user_message: str,\n'
    '        model_id: str,\n'
    '        messages: List[dict],\n'
    '        body: dict,\n'
    "        __user__: Optional[dict] = None,\n"
    '        __event_emitter__=None,\n'
    '    ) -> Union[str, Generator, Iterator]:'
)

new_sig = (
    '    def pipe(\n'
    '        self,\n'
    '        body: dict,\n'
    "        __user__: Optional[dict] = None,\n"
    '        __event_emitter__=None,\n'
    '    ) -> Union[str, Generator, Iterator]:'
)

if old_sig in content:
    content = content.replace(old_sig, new_sig)
    print("OK: Pipe signature replaced")
else:
    print("WARN: Old signature not found!")
    idx = content.find('def pipe(')
    if idx >= 0:
        end = content.find(':', idx)
        print(f"  Found: {content[idx:end+1]}")

# Replace body extraction from messages
old_extract = (
    '        if not messages:\n'
    '            yield "No messages in request."\n'
    '            return\n'
    '\n'
    '        question = user_message or messages[-1].get("content", "")'
)

new_extract = (
    '        messages = body.get("messages", [])\n'
    '        if not messages:\n'
    '            yield "No messages in request."\n'
    '            return\n'
    '\n'
    '        user_message = messages[-1].get("content", "") if messages else ""\n'
    '        question = user_message'
)

if old_extract in content:
    content = content.replace(old_extract, new_extract)
    print("OK: Message extraction replaced")
else:
    print("WARN: Old extraction not found!")

# Write to SQLite
conn = sqlite3.connect(r'C:\Users\ivanr\AppData\Roaming\Python\Python311\site-packages\open_webui\data\webui.db')
cur = conn.cursor()
cur.execute('UPDATE function SET content = ?, is_active = 1 WHERE id = ?', (content, '5642eb48-f7d5-4e6f-bc8a-13bedfdcaf3f'))
print(f'Updated {cur.rowcount} row(s)')
conn.commit()

# Verify
cur.execute('SELECT is_active FROM function WHERE id = ?', ('5642eb48-f7d5-4e6f-bc8a-13bedfdcaf3f',))
row = cur.fetchone()
print(f'is_active={row[0]}')
conn.close()
print('Done!')
