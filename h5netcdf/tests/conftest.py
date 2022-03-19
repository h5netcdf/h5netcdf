import os
import sys
import tempfile
from pathlib import Path
from shutil import rmtree
import pytest
try:
    from h5pyd._apps.hstouch import main as hstouch
    from hsds.hsds_app import HsdsApp
    with_reqd_pkgs = True
except ImportError:
    with_reqd_pkgs = False


def set_hsds_root():
    """Make required HSDS root directory."""
    hsds_root = Path(os.environ['ROOT_DIR']) / os.environ['BUCKET_NAME'] / 'home'
    if hsds_root.exists():
        rmtree(hsds_root)

    old_sysargv = sys.argv
    sys.argv = ['']
    sys.argv.extend(['-e', os.environ['HS_ENDPOINT']])
    sys.argv.extend(['-u', 'admin'])
    sys.argv.extend(['-p', 'admin'])
    sys.argv.extend(['-b', os.environ['BUCKET_NAME']])
    sys.argv.append('/home/')
    hstouch()

    sys.argv = ['']
    sys.argv.extend(['-e', os.environ['HS_ENDPOINT']])
    sys.argv.extend(['-u', 'admin'])
    sys.argv.extend(['-p', 'admin'])
    sys.argv.extend(['-b', os.environ['BUCKET_NAME']])
    sys.argv.extend(['-o', os.environ['HS_USERNAME']])
    sys.argv.append(f'/home/{os.environ["HS_USERNAME"]}/')
    hstouch()
    sys.argv = old_sysargv


@pytest.fixture(scope='session')
def hsds_up():
    """Provide HDF Highly Scalabale Data Service (HSDS) for h5pyd testing."""
    if with_reqd_pkgs:
        root_dir = Path(tempfile.mkdtemp(prefix='tmp-hsds-root-'))
        os.environ['BUCKET_NAME'] = 'data'
        (root_dir / os.getenv('BUCKET_NAME')).mkdir(parents=True, exist_ok=True)
        os.environ['ROOT_DIR'] = str(root_dir)
        os.environ['HS_USERNAME'] = 'h5netcdf-pytest'
        os.environ['HS_PASSWORD'] = 'TestEarlyTestEverything'

        config = """allow_noauth: true
auth_expiration: -1
default_public: False
aws_access_key_id: xxx
aws_secret_access_key: xxx
aws_iam_role: hsds_role
aws_region: us-east-1
hsds_endpoint: http://hsds.hdf.test
aws_s3_gateway: null
aws_dynamodb_gateway: null
aws_dynamodb_users_table: null
azure_connection_string: null
azure_resource_id: null
azure_storage_account: null
azure_resource_group: null
root_dir: null
password_salt: null
bucket_name: hsdstest
head_port: 5100
head_ram: 512m
dn_port: 6101
dn_ram: 3g
sn_port: 5101
sn_ram: 1g
rangeget_port: 6900
rangeget_ram: 2g
target_sn_count: 0
target_dn_count: 0
log_level: INFO
log_timestamps: false
log_prefix: null
max_tcp_connections: 100
head_sleep_time: 10
node_sleep_time: 10
async_sleep_time: 10
s3_sync_interval: 1
s3_sync_task_timeout: 10
store_read_timeout: 1
store_read_sleep_interval: 0.1
max_pending_write_requests: 20
flush_sleep_interval: 1
max_chunks_per_request: 1000
min_chunk_size: 1m
max_chunk_size: 4m
max_request_size: 100m
max_chunks_per_folder: 0
max_task_count: 100
max_tasks_per_node_per_request: 16
aio_max_pool_connections: 64
metadata_mem_cache_size: 128m
metadata_mem_cache_expire: 3600
chunk_mem_cache_size: 128m
chunk_mem_cache_expire: 3600
data_cache_size: 128m
data_cache_max_req_size: 128k
data_cache_expire_time: 3600
data_cache_page_size: 4m
data_cache_max_concurrent_read: 16
timeout: 30
password_file: /config/passwd.txt
groups_file: /config/groups.txt
server_name: Highly Scalable Data Service (HSDS)
greeting: Welcome to HSDS!
about: HSDS is a webservice for HDF data
top_level_domains: []
cors_domain: "*"
admin_user: admin
admin_group: null
openid_provider: azure
openid_url: null
openid_audience: null
openid_claims: unique_name,appid,roles
chaos_die: 0
standalone_app: false
blosc_nthreads: 2
http_compression: false
http_max_url_length: 512
k8s_app_label: hsds
k8s_namespace: null
restart_policy: on-failure
domain_req_max_objects_limit: 500
"""
        tmp_dir = Path(tempfile.mkdtemp(prefix='tmp-hsds-'))
        config_file = tmp_dir / 'config.yml'
        config_file.write_text(config)
        passwd_file = tmp_dir / 'passwd.txt'
        passwd_file.write_text(
            f'admin:admin\n{os.environ["HS_USERNAME"]}:{os.environ["HS_PASSWORD"]}\n')
        log_file = str(tmp_dir / 'hsds.log')
        tmp_dir = str(tmp_dir)
        if sys.platform == 'darwin':
            # macOS temp directory paths can be very long and break low-level
            # socket comms code...
            socket_dir = '/tmp/hsds'
        else:
            socket_dir = tmp_dir

        try:
            hsds = HsdsApp(
                username=os.environ['HS_USERNAME'],
                password=os.environ['HS_PASSWORD'],
                password_file=str(passwd_file),
                log_level=os.getenv('LOG_LEVEL', 'DEBUG'),
                logfile=log_file,
                socket_dir=socket_dir,
                config_dir=tmp_dir,
                dn_count=2)
            hsds.run()
            is_up = hsds.ready

            if is_up:
                os.environ['HS_ENDPOINT'] = hsds.endpoint
                set_hsds_root()
        except Exception:
            is_up = False

        yield is_up

        hsds.stop()
        rmtree(tmp_dir, ignore_errors=True)
        rmtree(socket_dir, ignore_errors=True)
        rmtree(root_dir, ignore_errors=True)

    else:
        yield False
