<template>
    <v-card elevation="3" outlined :loading="isLoading_">
        <!-- v-list-items and v-slider ensures consistent padding for card; otherwise, the loading bar would
        cause a jump in spacing when loading vs not loading: -->
        <v-list>
            <v-list-item class="py-2 px-6">
                <v-list-item-content>
                    <v-row>
                        <!-- ----- Selected product image: ----- -->
                        <v-col cols="12" sm="4" lg="3" class="d-flex align-center justify-center justify-sm-start">
                            <v-avatar v-if="thumbnail" tile size="130">
                                <v-img :src="thumbnail" contain alt="selected product image" />
                            </v-avatar>
                            <v-img
                                v-else
                                contain
                                width="100"
                                max-width="100"
                                height="auto"
                                src="/package.svg"
                                alt="logo of package box"
                            />
                        </v-col>

                        <!-- ----- Drop zone: ----- -->
                        <v-col cols="12" sm="8" lg="9">
                            <div
                                class="file-upload__dropzone"
                                :class="{'file-upload__dropzone--over': isDragging}"
                                @dragenter="isDragging = true"
                                @dragleave="isDragging = false"
                            >
                                <div class="file-upload__info" @drag="uploadFile">
                                    <img class="file-upload__icon" src="/upload_file.svg" alt="upload icon">
                                    <div class="pt-2">
                                        <span>Drag image here</span>
                                        <a>or browse</a>
                                    </div>
                                </div>
                                <input ref="inputUpload" class="file-upload__input" type="file" @change="uploadFile">
                            </div>
                        </v-col>
                    </v-row>
                </v-list-item-content>
            </v-list-item>
        </v-list>
    </v-card>
</template>

<script>
import axios from 'axios'

export default {
    name: 'FileUpload',
    props: {
        isLoading: {
            type: Boolean,
            default: false
        },
        thumbnail: {
            type: String,
            default: ''
        }
    },
    data() {
        return {
            isDragging: false,
            isLoading_: false
        }
    },
    watch: {
        isLoading() {
            this.isLoading_ = this.isLoading
        }
    },
    methods: {
        uploadFile(e) {
            this.isDragging = false

            const files = e.target.files || e.dataTransfer.files
            const file = files?.[0]

            // Validate uploaded file:
            if(!file) {
                return
            } else if(!['image/jpeg', 'image/png'].includes(file.type)) {
                this.$emit('error', 'Please select a valid image file (jpg or png)')
                return
            } else if(file.size > 5000000) { // Limit file size to 5MB
                this.$emit('error', 'Please check file size no over 5 MB')
                return
            }

            this.isLoading_ = true
            this.$emit('error', '')

            // N.b. we need to use `FormData` and set `Content-Type` in order for API to handle file;
            // see <https://stackoverflow.com/q/43013858>:
            const formData = new FormData()
            formData.append('file', file)
            // TODO: Technically, this should be two API calls: PUT to put the resource on the server,
            // and GET to fetch the results
            axios.put(
                'http://0.0.0.0:5000/api/file-upload',
                formData,
                {
                    headers: {
                        'Content-Type': 'multipart/form-data'
                    }
                })
                .then((response) => {
                    // if error response:
                    if(response?.data?.category === 'error') {
                        const errorMessage = response?.data?.message || 'Invalid file type: only jpg and png files are accepted.'
                        this.$emit('error', errorMessage)
                    // if successful response:
                    } else {
                        this.$emit('get-products', {
                            'file-paths': response?.data?.file_paths
                        })
                        this.$emit('select-img', file)
                    }
                })
                .catch((error) => {
                    this.$emit('error', error)
                })
                .finally(() => {
                    this.isLoading_ = false
                })
        },
        removeFile() {
            this.$emit('select-img', '')
            // We need to reset the input (i.e. remove the uploaded file from input); otherwise, if user uploads an
            // image, calls `removeFile` method and trys uploading the same image, nothing will happen. If it's the same
            // image, upload handler won't trigger at all; hence the reason for clearing the input:
            this.$refs.inputUpload.value = ''
        }
    }
}
</script>

<style scoped lang="scss">
    .file-upload {
        &__dropzone {
            flex-grow: 1;
            height: 130px;
            position: relative;
            border-radius: 5px;
            border: 3px dashed $color-gray;
            transition: 200ms ease-in-out border-color;

            @media screen and (max-width: 599px) {
                height: 110px;
            }

            &:hover,
            &.file-upload__dropzone--over {
                border-color: darken($color-gray, 15%);
            }
        }

        &__input {
            position: absolute;
            cursor: pointer;
            top: 0;
            right: 0;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 0;
        }

        &__info {
            color: #A8A8A8;
            position: absolute;
            top: 50%;
            width: 100%;
            transform: translate(0, -50%);
            text-align: center;
        }

        &__icon {
            height: auto;
            opacity: 0.3;
        }
    }
</style>
